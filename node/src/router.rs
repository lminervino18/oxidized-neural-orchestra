use std::{io, time::Duration};

use comms::{
    Acceptor, Connection, Connector, NetRtp, OrchEvent, OrchHandle, PullSpecResponse,
    build_reliable_transport, get_dataset_cursor,
    protocol::Entity,
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use log::{info, warn};
use parameter_server::service::ServerBuilder;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use worker::builder::WorkerBuilder;

/// Routes the first orchestrator connection to the appropriate runtime role.
///
/// A single `NodeRouter` represents one process lifecycle. After receiving a
/// [`comms::protocol::NodeSpec`] from the orchestrator, the process commits to
/// either the parameter-server or worker role for the duration of its run.
pub struct NodeRouter<F>
where
    F: AsyncFnMut() -> io::Result<(OwnedReadHalf, OwnedWriteHalf)>,
{
    acceptor: Acceptor<OwnedReadHalf, OwnedWriteHalf, F>,
}

impl<F> NodeRouter<F>
where
    F: AsyncFnMut() -> io::Result<(OwnedReadHalf, OwnedWriteHalf)>,
{
    /// Creates a new `NodeRouter`.
    ///
    /// # Args
    /// * `acceptor` - The acceptor used to receive incoming connections.
    ///
    /// # Returns
    /// A new `NodeRouter` instance.
    pub fn new(acceptor: Acceptor<OwnedReadHalf, OwnedWriteHalf, F>) -> Self {
        Self { acceptor }
    }

    /// Accepts the orchestrator connection, reads the bootstrap spec, and runs the assigned role.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn run(mut self) -> io::Result<()> {
        let Connection::Orchestrator(mut orch_handle) = self.acceptor.accept().await? else {
            return Err(io::Error::other(
                "expected orchestrator as first connection",
            ));
        };

        match orch_handle.pull_specification().await? {
            PullSpecResponse::ParameterServer(spec) => {
                run_server(self.acceptor, orch_handle, spec).await
            }
            PullSpecResponse::Worker(spec) => run_worker(orch_handle, spec).await,
        }
    }
}

async fn run_server<F>(
    mut acceptor: Acceptor<OwnedReadHalf, OwnedWriteHalf, F>,
    mut orch_handle: OrchHandle<NetRtp>,
    spec: ServerSpec,
) -> io::Result<()>
where
    F: AsyncFnMut() -> io::Result<(OwnedReadHalf, OwnedWriteHalf)>,
{
    let nworkers = spec.nworkers;
    info!("node bootstrapped as parameter server, expecting {nworkers} worker(s)");

    let mut pserver = ServerBuilder::new()
        .build::<OwnedReadHalf, OwnedWriteHalf>(spec)
        .map_err(io::Error::other)?;

    for i in 0..nworkers {
        let conn = acceptor.accept().await?;
        let Connection::Worker(worker_handle) = conn else {
            return Err(io::Error::other(format!(
                "expected worker connection {}/{nworkers}, got unexpected connection type",
                i + 1
            )));
        };

        info!("worker {}/{nworkers} connected", i + 1);
        pserver.spawn(worker_handle);
    }

    let mut params = pserver.run().await?;
    serve_params(&mut orch_handle, &mut params).await
}

async fn run_worker(mut orch_handle: OrchHandle<NetRtp>, spec: WorkerSpec) -> io::Result<()> {
    let x_size_bytes = spec.dataset.x_size_bytes as usize;
    let y_size_bytes = spec.dataset.y_size_bytes as usize;
    let mut samples_raw = vec![0f32; x_size_bytes / size_of::<f32>()];
    let mut labels_raw = vec![0f32; y_size_bytes / size_of::<f32>()];

    orch_handle
        .pull_dataset(
            &mut get_dataset_cursor(&mut samples_raw),
            &mut get_dataset_cursor(&mut labels_raw),
            x_size_bytes,
            y_size_bytes,
        )
        .await?;

    let connector = Connector::new(
        |rx, tx| {
            build_reliable_transport(rx, tx, Duration::from_secs(5), Duration::from_secs(2), 2, 5)
        },
        Entity::Worker { id: spec.worker_id },
    );

    let mut worker = WorkerBuilder::new()
        .build(spec, connector, orch_handle, samples_raw, labels_raw)
        .await?;

    worker.run().await
}

async fn serve_params(
    orch_handle: &mut OrchHandle<NetRtp>,
    params: &mut Vec<f32>,
) -> io::Result<()> {
    info!("training complete, sending parameters");
    loop {
        match orch_handle.recv_event().await? {
            OrchEvent::RequestParams => orch_handle.push_params(params).await?,
            OrchEvent::Disconnect => {
                orch_handle.disconnect().await?;
                info!("disconnected");
                return Ok(());
            }
            OrchEvent::Stop => warn!("unexpected stop event during parameter serving"),
        }
    }
}
