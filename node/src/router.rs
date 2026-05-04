use std::io;

use comms::{
    Acceptor, Connection, Connector, OrchEvent, OrchHandle, PullSpecResponse, TransportLayer,
    get_dataset_cursor,
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
pub struct NodeRouter<T, F, G>
where
    T: TransportLayer,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(OwnedReadHalf, OwnedWriteHalf) -> T,
{
    acceptor: Acceptor<T, F>,
    connector: Connector<OwnedReadHalf, OwnedWriteHalf, T, G>,
}

impl<T, F, G> NodeRouter<T, F, G>
where
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(OwnedReadHalf, OwnedWriteHalf) -> T + Clone,
{
    /// Creates a new `NodeRouter`.
    ///
    /// # Args
    /// * `acceptor` - The acceptor used to receive incoming connections.
    /// * `connector` - The connector used by workers to reach parameter servers.
    ///
    /// # Returns
    /// A new `NodeRouter` instance.
    pub fn new(
        acceptor: Acceptor<T, F>,
        connector: Connector<OwnedReadHalf, OwnedWriteHalf, T, G>,
    ) -> Self {
        Self {
            acceptor,
            connector,
        }
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

            match run_session(&mut self.acceptor, &self.connector, orch_handle).await {
                Ok(()) => info!("session finished, waiting for next session"),
                Err(e) => warn!("session failed: {e}, resetting and waiting for next session"),
            }
        }
    }
}

async fn run_session<T, F, G>(
    acceptor: &mut Acceptor<T, F>,
    connector: &Connector<OwnedReadHalf, OwnedWriteHalf, T, G>,
    mut orch_handle: OrchHandle<T>,
) -> io::Result<()>
where
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(OwnedReadHalf, OwnedWriteHalf) -> T + Clone,
{
    match orch_handle.pull_specification().await? {
        PullSpecResponse::ParameterServer(spec) => {
            info!("starting parameter server session");
            run_server(acceptor, orch_handle, spec).await?;
            info!("parameter server session finished");
            Ok(())
        }
        PullSpecResponse::Worker(spec) => {
            info!("starting worker session");
            run_worker(acceptor, connector, orch_handle, spec).await?;
            info!("worker session finished");
            Ok(())
        }
    }
}

async fn run_server<T, F>(
    acceptor: &mut Acceptor<T, F>,
    mut orch_handle: OrchHandle<T>,
    spec: ServerSpec,
) -> io::Result<()>
where
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
{
    let mut server_builder = ServerBuilder::new(acceptor);
    let mut pserver = server_builder.build(spec).await?;

    let mut params = pserver.run().await?;
    loop {
        match orch_handle.recv_event().await? {
            OrchEvent::Disconnect => {
                orch_handle.disconnect().await?;
                break;
            }
            OrchEvent::RequestParams => orch_handle.push_params(&mut params).await?,
            event => warn!("Unexpected OrchEvent: {event:?}"),
        }
    }

    Ok(())
}

async fn run_worker<T, F, G>(
    acceptor: &mut Acceptor<T, F>,
    connector: &Connector<OwnedReadHalf, OwnedWriteHalf, T, G>,
    mut orch_handle: OrchHandle<T>,
    spec: WorkerSpec,
) -> io::Result<()>
where
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(OwnedReadHalf, OwnedWriteHalf) -> T + Clone,
{
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

    let mut worker = WorkerBuilder::new(acceptor, connector.clone())
        .build(spec, orch_handle, samples_raw, labels_raw)
        .await?;

    worker.run().await
}
