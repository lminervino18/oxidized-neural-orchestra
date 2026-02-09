use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
    specs::worker::{AlgorithmSpec, LossReportSpec},
};
use log::{debug, info, warn};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::TcpStream,
};

use crate::error::WorkerError;
use machine_learning::training::Trainer;

/// Infrastructure worker runtime.
pub struct Worker {
    worker_id: usize,
    algorithm: AlgorithmSpec,
    loss_report: LossReportSpec,
    trainer: Box<dyn Trainer>,
}

impl Worker {
    /// Worker runtime that maps weights snapshots into gradient updates.
    ///
    /// # Args
    /// * `worker_id` - Identifier used for observability.
    /// * `algorithm` - Distributed algorithm selection for this worker.
    /// * `loss_report` - Loss reporting policy used to emit epoch telemetry to the orchestrator.
    /// * `trainer` - Domain strategy used to compute gradients from weights.
    ///
    /// # Returns
    /// A new worker instance.
    pub fn new(
        worker_id: usize,
        algorithm: AlgorithmSpec,
        loss_report: LossReportSpec,
        trainer: Box<dyn Trainer>,
    ) -> Self {
        Self {
            worker_id,
            algorithm,
            loss_report,
            trainer,
        }
    }

    /// Runs the worker using its configured distributed algorithm while keeping a live
    /// bidirectional channel to the orchestrator.
    ///
    /// # Args
    /// * `orch_rx` - Receiving end of the orchestrator channel.
    /// * `orch_tx` - Sending end of the orchestrator channel.
    ///
    /// # Returns
    /// Returns `Ok(())` on graceful completion.
    ///
    /// # Errors
    /// Returns `WorkerError` on I/O failures or protocol violations.
    pub async fn run<Rorch, Worch>(
        self,
        orch_rx: OnoReceiver<Rorch>,
        orch_tx: OnoSender<Worch>,
    ) -> Result<(), WorkerError>
    where
        Rorch: AsyncRead + Unpin + Send,
        Worch: AsyncWrite + Unpin + Send,
    {
        match self.algorithm {
            AlgorithmSpec::ParameterServer { server_ip } => {
                info!(
                    "connecting to parameter server: worker_id={} server_addr={}",
                    self.worker_id, server_ip
                );

                let ps_stream = TcpStream::connect(server_ip)
                    .await
                    .map_err(WorkerError::Io)?;
                let (ps_rx, ps_tx) = ps_stream.into_split();
                let (ps_rx, ps_tx) = comms::channel(ps_rx, ps_tx);

                self.run_parameter_server(ps_rx, ps_tx, orch_rx, orch_tx).await
            }
        }
    }

    /// Runs the parameter-server protocol loop while listening to the orchestrator control plane.
    ///
    /// This loop provides:
    /// - Data-plane: weights/gradients exchange with the parameter server.
    /// - Control-plane: early-stop via orchestrator `Disconnect`, and epoch telemetry via `ReportLosses`.
    /// # Args
    /// * `ps_rx` - Receiving end of the parameter-server channel.
    /// * `ps_tx` - Sending end of the parameter-server channel.
    /// * `orch_rx` - Receiving end of the orchestrator channel.
    /// * `orch_tx` - Sending end of the orchestrator channel.
    ///
    /// # Errors
    /// Returns `WorkerError` on I/O failures or protocol invariant violations.
    pub async fn run_parameter_server<Rps, Wps, Rorch, Worch>(
        self,
        mut ps_rx: OnoReceiver<Rps>,
        mut ps_tx: OnoSender<Wps>,
        mut orch_rx: OnoReceiver<Rorch>,
        mut orch_tx: OnoSender<Worch>,
    ) -> Result<(), WorkerError>
    where
        Rps: AsyncRead + Unpin + Send,
        Wps: AsyncWrite + Unpin + Send,
        Rorch: AsyncRead + Unpin + Send,
        Worch: AsyncWrite + Unpin + Send,
    {
        let worker_id = self.worker_id;
        let loss_report = self.loss_report;
        let mut trainer = self.trainer;

        let mut step: usize = 0;

        info!("worker starting: worker_id={worker_id}");

        loop {
            debug!("waiting message: worker_id={worker_id} step={step}");

            tokio::select! {
                ps_msg = ps_rx.recv::<Msg>() => {
                    let msg = ps_msg.map_err(WorkerError::Io)?;

                    match msg {
                        Msg::Control(Command::Disconnect) => {
                            info!("disconnect received from parameter server: worker_id={worker_id} step={step}");
                            break;
                        }

                        Msg::Data(Payload::Params(weights)) => {
                            let got = weights.len();
                            debug!("training: worker_id={worker_id} step={step} params={got}");

                            let grad = trainer.train(weights);

                            ps_tx.send(&Msg::Data(Payload::Grad(grad)))
                                .await
                                .map_err(WorkerError::Io)?;

                            step += 1;

                            let epoch = step;
                            if should_report_loss(loss_report, epoch) {

                                //Here i send fake losses, the training domain will need to be extended to compute real epoch losses and report them here.
                                orch_tx
                                    .send(&Msg::Control(Command::ReportLosses {
                                        worker_id,
                                        epoch,
                                        losses: Vec::new(),
                                    }))
                                    .await
                                    .map_err(WorkerError::Io)?;
                            }
                        }

                        other => {
                            warn!(
                                "unexpected message from parameter server: worker_id={} step={} got={}",
                                worker_id,
                                step,
                                msg_kind(&other)
                            );

                            return Err(WorkerError::UnexpectedMessage {
                                step,
                                got: msg_kind(&other),
                            });
                        }
                    }
                }

                orch_msg = orch_rx.recv::<Msg>() => {
                    let msg = orch_msg.map_err(WorkerError::Io)?;

                    match msg {
                        Msg::Control(Command::Disconnect) => {
                            info!("disconnect received from orchestrator: worker_id={worker_id} step={step}");
                            break;
                        }
                        other => {
                            warn!(
                                "unexpected message from orchestrator: worker_id={} step={} got={}",
                                worker_id,
                                step,
                                msg_kind(&other),
                            );
                            // Ignore unknown control-plane messages for now.
                        }
                    }
                }
            }
        }

        info!("worker finished: worker_id={worker_id} steps={step}");
        Ok(())
    }
}

fn should_report_loss(policy: LossReportSpec, epoch: usize) -> bool {
    match policy {
        LossReportSpec::Disabled => false,
        LossReportSpec::EveryEpoch => true,
        LossReportSpec::EveryNEpochs { n } => n != 0 && epoch % n == 0,
    }
}

fn msg_kind(msg: &Msg<'_>) -> &'static str {
    match msg {
        Msg::Control(_) => "control",
        Msg::Err(_) => "err",
        Msg::Data(Payload::Grad(_)) => "data/gradient",
        Msg::Data(Payload::Params(_)) => "data/weights",
    }
}
