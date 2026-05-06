use std::io;

use comms::{OrchEvent, OrchHandle, TransportLayer};
use log::{debug, info, warn};
use machine_learning::training::{TrainResult, Trainer};

use crate::{middlewares::WorkerRingManager, workers::Worker};

/// The middleman between the workers and the model trainer.
pub struct AllReduceWorker<T>
where
    T: TransportLayer,
{
    trainer: Box<dyn Trainer>,
    ring_manager: WorkerRingManager<T>,
    orch_handle: OrchHandle<T>,
    optimization_params: Vec<f32>,
    params: Vec<f32>,
}

impl<T> AllReduceWorker<T>
where
    T: TransportLayer,
{
    /// Creates a new `AllReduceWorker`.
    ///
    /// # Args
    /// * `trainer` - Domain strategy used to compute gradients from weights.
    /// * `cluster_manager` - The manager for communicating with the server cluster.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    /// * `params` - The initial parameters of the model.
    ///
    /// # Returns
    /// A new `AllReduceWorker` instance.
    pub fn new(
        trainer: Box<dyn Trainer>,
        ring_manager: WorkerRingManager<T>,
        orch_handle: OrchHandle<T>,
        params: Vec<f32>,
    ) -> Self {
        Self {
            trainer,
            ring_manager,
            orch_handle,
            optimization_params: params.clone(),
            params,
        }
    }
}

#[async_trait::async_trait]
impl<T> Worker for AllReduceWorker<T>
where
    T: TransportLayer,
{
    async fn run(&mut self) -> io::Result<()> {
        let mut should_continue = true;

        while should_continue {
            let mut param_manager = self
                .ring_manager
                .build_param_manager(&mut self.optimization_params);

            let TrainResult { losses, was_last } = self.trainer.train(&mut param_manager).unwrap();
            self.orch_handle.push_losses(losses).await?;
            should_continue = !was_last;

            tokio::select! {
                biased;
                event = self.orch_handle.recv_event() => match event? {
                    OrchEvent::Stop => {
                        info!("received a stop command from orchestrator");
                        break;
                    }
                    OrchEvent::Disconnect => {
                        info!("received a disconnect command from the orchestrator");
                        break;
                    }
                    other => {
                        warn!("unexpected message from orchestrator, got: {other:?}");
                    }
                },
                response = self.ring_manager.pull_grads(&mut self.params) => {
                    debug!("received gradients from all workers, training...");

                    let mut param_manager = response?;

                    // SAFETY: The parameter and gradient buffer have the same size.
                    self.trainer.optimize(&mut param_manager).unwrap();
                    param_manager.zero_grad();

                    self.optimization_params.copy_from_slice(&self.params);
                }
            }
        }

        self.orch_handle.done().await?;

        if let OrchEvent::RequestParams = self.orch_handle.recv_event().await? {
            self.orch_handle.push_params(&mut self.params).await?;
        }

        self.orch_handle.disconnect().await?;
        self.ring_manager.disconnect().await?;
        Ok(())
    }
}
