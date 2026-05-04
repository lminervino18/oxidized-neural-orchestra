use std::io;

use comms::{OrchEvent, OrchHandle, TransportLayer};
use log::{debug, info, warn};
use machine_learning::training::{TrainResult, Trainer};

use crate::{middlewares::ServerClusterManager, workers::Worker};

/// The middleman between the parameter server and the model trainer.
pub struct ParamServerWorker<T>
where
    T: TransportLayer,
{
    trainer: Box<dyn Trainer>,
    cluster_manager: ServerClusterManager<T>,
    orch_handle: OrchHandle<T>,
}

impl<T> ParamServerWorker<T>
where
    T: TransportLayer,
{
    /// Creates a new `Worker`.
    ///
    /// # Args
    /// * `trainer` - Domain strategy used to compute gradients from weights.
    /// * `cluster_manager` - The manager for communicating with the server cluster.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    ///
    /// # Returns
    /// A new `Worker` instance.
    pub fn new(
        trainer: Box<dyn Trainer>,
        cluster_manager: ServerClusterManager<T>,
        orch_handle: OrchHandle<T>,
    ) -> Self {
        Self {
            trainer,
            cluster_manager,
            orch_handle,
        }
    }
}

#[async_trait::async_trait]
impl<T> Worker for ParamServerWorker<T>
where
    T: TransportLayer,
{
    /// Runs the worker using its configured distributed algorithm while keeping a live
    /// bidirectional channel to the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn run(&mut self) -> io::Result<()> {
        let mut should_continue = true;

        while should_continue {
            tokio::select! {
                biased;
                event = self.orch_handle.recv_event() => match event? {
                    OrchEvent::Stop => {
                        info!("received a stop command from orchestrator");
                        should_continue = false;
                    }
                    OrchEvent::Disconnect => {
                        info!("received a disconnect command from the orchestrator");
                        break;
                    }
                    other => {
                        warn!("unexpected message from orchestrator, got: {other:?}");
                    }
                },
                response = self.cluster_manager.pull_params() => {
                    debug!("received parameters from all servers, training...");

                    let mut param_manager = response?;
                    let TrainResult { losses, was_last } = self.trainer.train(&mut param_manager).unwrap();
                    self.cluster_manager.push_grads().await?;

                    self.orch_handle.push_losses(losses).await?;
                    should_continue = !was_last;
                }
            }
        }

        self.cluster_manager.disconnect().await?;
        self.orch_handle.disconnect().await?;
        Ok(())
    }
}
