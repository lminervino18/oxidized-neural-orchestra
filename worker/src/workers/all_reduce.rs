use std::io;

use comms::{OrchEvent, OrchHandle, TransportLayer, specs::machine_learning::ParamGenSpec};
use log::{debug, info, warn};
use machine_learning::training::{TrainResult, Trainer};

use super::{Run, Worker};
use crate::middlewares::WorkerRingManager;

/// The middleman between the workers and the model trainer.
pub struct AllReduceWorker<'node, T>
where
    T: TransportLayer,
{
    trainer: Box<dyn Trainer>,
    ring_manager: WorkerRingManager<T>,
    orch_handle: &'node mut OrchHandle<T>,
    optimization_params: Vec<f32>,
    params: Vec<f32>,
}

impl<'node, T> AllReduceWorker<'node, T>
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
        orch_handle: &'node mut OrchHandle<T>,
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
impl<T> Worker for AllReduceWorker<'_, T>
where
    T: TransportLayer,
{
    async fn run(&mut self) -> io::Result<Run> {
        let mut should_continue = true;

        while should_continue {
            let mut param_manager = self
                .ring_manager
                .build_param_manager(&mut self.optimization_params);

            let TrainResult { losses, was_last } = self
                .trainer
                .train(&mut param_manager)
                .map_err(io::Error::other)?;

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
                    OrchEvent::Upgrade { mut spec, ranges } => {
                        info!("upgrading to parameter server");

                        let params: Vec<_> = ranges
                            .into_iter()
                            .flat_map(|(a, b)| &self.params[a..b])
                            .cloned()
                            .collect();

                        spec.param_gen = ParamGenSpec::Inline { params };
                        return Ok(Run::Upgrade { spec })
                    }
                    OrchEvent::Switch {
                        server_addrs,
                        server_sizes,
                        server_ordering,
                        trainer_spec,
                    } => {
                        info!("switching algorithm to parameter server as a worker");
                        return Ok(Run::Switch {
                            server_addrs,
                            server_sizes,
                            server_ordering,
                            trainer_spec,
                        });
                    }
                    other => {
                        warn!("unexpected message from orchestrator, got: {other:?}");
                    }
                },
                response = self.ring_manager.pull_grads(&mut self.params) => {
                    debug!("received gradients from all workers, training...");

                    let Ok(mut param_manager) = response else {
                        continue;
                    };

                    // SAFETY: The parameter and gradient buffer have the same size.
                    self.trainer.optimize(&mut param_manager).unwrap();
                    param_manager.zero_grad();

                    self.optimization_params.copy_from_slice(&self.params);
                }
            }
        }

        self.orch_handle.done().await?;
        debug!("sent done to orchestrator");

        loop {
            debug!("waiting on orchestrator event");
            let event = self.orch_handle.recv_event().await?;
            debug!("received {event:?} from orchestrator");

            match event {
                OrchEvent::Disconnect => break,
                OrchEvent::RequestParams => self.orch_handle.push_params(&mut self.params).await?,
                other => warn!("unexpected message from orchestrator, got: {other:?}"),
            }
        }

        Ok(Run::Done)
    }

    fn into_trainer(self: Box<Self>) -> Box<dyn Trainer> {
        self.trainer
    }
}
