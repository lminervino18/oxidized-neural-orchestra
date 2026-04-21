use std::io;

use comms::{OrchHandle, TransportLayer};
use log::debug;
use machine_learning::training::{TrainResult, Trainer};

use crate::cluster_manager::ServerClusterManager;

/// The middleman between the parameter server and the model trainer.
pub struct Worker {
    trainer: Box<dyn Trainer>,
}

impl Worker {
    /// Creates a new `Worker`.
    ///
    /// # Args
    /// * `trainer` - Domain strategy used to compute gradients from weights.
    ///
    /// # Returns
    /// A new `Worker` instance.
    pub fn new(trainer: Box<dyn Trainer>) -> Self {
        Self { trainer }
    }

    /// Runs the worker using its configured distributed algorithm while keeping a live
    /// bidirectional channel to the orchestrator.
    ///
    /// # Args
    /// * `rx` - The receiving end of the communication between the worker and the orchestrator.
    /// * `tx` - The sending end of the communication between the worker and the orchestrator.
    /// * `middleware` - The communication manager between this worker and the parameter servers.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn run<T>(
        self,
        mut orch_handle: OrchHandle<T>,
        mut cluster_manager: ServerClusterManager<T>,
    ) -> io::Result<()>
    where
        T: TransportLayer,
    {
        let mut trainer = self.trainer;
        let mut should_continue = true;

        while should_continue {
            let mut param_manager = cluster_manager.pull_params().await?;
            debug!("received parameters from all servers, training...");

            let TrainResult { losses, was_last } = trainer.train(&mut param_manager).unwrap();
            cluster_manager.push_grads().await?;

            orch_handle.push_losses(losses).await?;
            should_continue = !was_last;
        }

        cluster_manager.disconnect().await?;
        orch_handle.disconnect().await?;
        Ok(())
    }
}
