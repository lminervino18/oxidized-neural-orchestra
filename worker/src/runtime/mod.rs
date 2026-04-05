pub mod ps;
pub mod ring;

use std::{future::Future, io, pin::Pin};

use comms::specs::{algorithm::AlgorithmSpec, worker::WorkerSpec};
use tokio::net::{
    TcpListener,
    tcp::{OwnedReadHalf, OwnedWriteHalf},
};

use crate::builder::WorkerBuilder;

pub type OrchRx = comms::OnoReceiver<OwnedReadHalf>;
pub type OrchTx = comms::OnoSender<OwnedWriteHalf>;
pub type RuntimeFuture = Pin<Box<dyn Future<Output = io::Result<()>> + 'static>>;

/// The executable distributed runtime for a worker process.
pub trait DistributedRuntime {
    /// Runs the distributed runtime until completion.
    ///
    /// # Returns
    /// An io error if the runtime fails.
    fn run(self: Box<Self>) -> RuntimeFuture;
}

/// Builds the distributed runtime for a worker according to its configured algorithm.
///
/// # Args
/// * `spec` - The worker specification received from the orchestrator.
/// * `dataset_raw` - The raw dataset partition assigned to this worker.
/// * `orch_rx` - The receiving end of the communication between the worker and the orchestrator.
/// * `orch_tx` - The sending end of the communication between the worker and the orchestrator.
/// * `listener` - The worker listener used to accept ring-neighbor connections when needed.
///
/// # Returns
/// The boxed distributed runtime for this worker.
///
/// # Errors
/// Returns an io error if the runtime bootstrap fails.
pub async fn build(
    spec: WorkerSpec,
    dataset_raw: Vec<f32>,
    orch_rx: OrchRx,
    orch_tx: OrchTx,
    listener: TcpListener,
) -> io::Result<Box<dyn DistributedRuntime>> {
    match spec.algorithm.clone() {
        AlgorithmSpec::ParameterServer(ps_spec) => {
            let serializer = spec.serializer.clone();
            let worker_builder = WorkerBuilder::new();
            let worker =
                worker_builder.build_parameter_server(spec, &ps_spec.server_sizes, dataset_raw);

            let runtime = ps::ParameterServerRuntime::bootstrap(
                worker, ps_spec, serializer, orch_rx, orch_tx,
            )
            .await?;

            Ok(Box::new(runtime) as Box<dyn DistributedRuntime>)
        }
        AlgorithmSpec::RingAllReduce(ring_spec) => {
            let worker_builder = WorkerBuilder::new();
            let worker = worker_builder.build_ring_all_reduce(spec.clone(), dataset_raw);

            let runtime = ring::RingAllReduceRuntime::bootstrap(
                worker,
                spec.worker_id,
                ring_spec,
                orch_rx,
                orch_tx,
                listener,
            )
            .await?;

            Ok(Box::new(runtime) as Box<dyn DistributedRuntime>)
        }
    }
}
