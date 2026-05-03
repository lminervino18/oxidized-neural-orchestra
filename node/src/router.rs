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

/// Routes incoming orchestrator connections to the appropriate runtime role,
/// keeping the process alive across sequential sessions.
///
/// Each session is isolated: state is owned within the session stack frame and
/// dropped naturally when the session ends. The process then loops back to
/// accept the next session.
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

    /// Runs the sequential session loop.
    ///
    /// Blocks indefinitely, accepting one orchestrator connection per iteration.
    /// Session-level errors are logged and the loop continues. Returns only if
    /// the listener itself fails with a fatal error.
    ///
    /// # Errors
    /// Returns an io error if the underlying listener fails.
    pub async fn run(mut self) -> io::Result<()> {
        loop {
            info!("waiting for next session");

            let conn = self.acceptor.accept().await?;

            let Connection::Orchestrator(orch_handle) = conn else {
                warn!(
                    "expected orchestrator as first connection, got unexpected connection type; skipping"
                );
                continue;
            };

            info!("accepted orchestrator connection");

            match run_session(&mut self.acceptor, orch_handle).await {
                Ok(()) => info!("session finished, waiting for next session"),
                Err(e) => warn!("session failed: {e}, resetting and waiting for next session"),
            }
        }
    }
}

async fn run_session<F>(
    acceptor: &mut Acceptor<OwnedReadHalf, OwnedWriteHalf, F>,
    mut orch_handle: OrchHandle<NetRtp>,
) -> io::Result<()>
where
    F: AsyncFnMut() -> io::Result<(OwnedReadHalf, OwnedWriteHalf)>,
{
    match orch_handle.pull_specification().await? {
        PullSpecResponse::ParameterServer(spec) => {
            info!("starting parameter server session server_id={}", spec.id);
            run_server(acceptor, orch_handle, spec).await?;
            info!("parameter server session finished");
            Ok(())
        }
        PullSpecResponse::Worker(spec) => {
            info!("starting worker session worker_id={}", spec.worker_id);
            run_worker(orch_handle, spec).await?;
            info!("worker session finished");
            Ok(())
        }
    }
}

async fn run_server<F>(
    acceptor: &mut Acceptor<OwnedReadHalf, OwnedWriteHalf, F>,
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
