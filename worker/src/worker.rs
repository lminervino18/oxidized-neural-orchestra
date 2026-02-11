use std::num::NonZeroUsize;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
    specs::worker::AlgorithmSpec,
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
    max_epochs: NonZeroUsize,
    algorithm: AlgorithmSpec,
    trainer: Box<dyn Trainer>,
}

impl Worker {
    /// Worker runtime that maps weights snapshots into gradient updates.
    ///
    /// # Args
    /// * `worker_id` - Identifier used for observability.
    /// * `max_iters` - The maximum amount of iterations to run.
    /// * `algorithm` - Distributed algorithm selection for this worker.
    /// * `trainer` - Domain strategy used to compute gradients from weights.
    ///
    /// # Returns
    /// A new worker instance.
    pub fn new(
        worker_id: usize,
        max_iters: NonZeroUsize,
        algorithm: AlgorithmSpec,
        trainer: Box<dyn Trainer>,
    ) -> Self {
        Self {
            worker_id,
            max_epochs: max_iters,
            algorithm,
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
            AlgorithmSpec::ParameterServer { ref server_addr } => {
                info!(
                    "connecting to parameter server: worker_id={} server_addr={}",
                    self.worker_id, server_addr
                );

                let ps_stream = TcpStream::connect(server_addr).await?;
                let (ps_rx, ps_tx) = ps_stream.into_split();
                let (ps_rx, ps_tx) = comms::channel(ps_rx, ps_tx);

                self.run_parameter_server(ps_rx, ps_tx, orch_rx, orch_tx)
                    .await
            }
        }
    }

    /// Runs the parameter-server protocol loop while listening to the orchestrator control plane.
    ///
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
        let mut trainer = self.trainer;
        let mut epoch = 0;

        info!("worker starting: worker_id={worker_id}");

        while epoch < self.max_epochs.get() {
            debug!("waiting message: worker_id={worker_id} epoch={epoch}");

            tokio::select! {
                ps_msg = ps_rx.recv::<Msg>() => {
                    let msg = ps_msg?;
                    if handle_server_message(
                        worker_id,
                        &mut epoch,
                        &mut trainer,
                        &mut ps_tx,
                        &mut orch_tx,
                        msg,
                    ).await? {
                        break;
                    }
                }

                orch_msg = orch_rx.recv::<Msg>() => {
                    let msg = orch_msg?;
                    if handle_orchestrator_message(worker_id, epoch, msg) {
                        break;
                    }
                }
            }
        }

        info!("worker finished: worker_id={worker_id} epoch={epoch}");
        let msg = Msg::Control(Command::Disconnect);
        orch_tx.send(&msg).await?;

        Ok(())
    }
}

async fn handle_server_message<Wps, Worch>(
    worker_id: usize,
    epoch: &mut usize,
    trainer: &mut Box<dyn Trainer>,
    ps_tx: &mut OnoSender<Wps>,
    orch_tx: &mut OnoSender<Worch>,
    msg: Msg<'_>,
) -> Result<bool, WorkerError>
where
    Wps: AsyncWrite + Unpin + Send,
    Worch: AsyncWrite + Unpin + Send,
{
    match msg {
        Msg::Control(Command::Disconnect) => {
            info!("disconnect received from parameter server: worker_id={worker_id} epoch={epoch}");
            Ok(true)
        }

        Msg::Data(Payload::Params(weights)) => {
            let got = weights.len();
            debug!("training: worker_id={worker_id} epoch={epoch} params={got}");

            let (grad, losses) = trainer.train(weights);
            *epoch += losses.len();

            let mut msg = Msg::Data(Payload::Grad(grad));
            ps_tx.send(&msg).await?;

            msg = Msg::Control(Command::ReportLoss { worker_id, losses });
            orch_tx.send(&msg).await?;

            Ok(false)
        }

        other => {
            warn!(
                "unexpected message from parameter server: worker_id={} epoch={} got={}",
                worker_id,
                epoch,
                msg_kind(&other)
            );

            Err(WorkerError::UnexpectedMessage {
                epoch: *epoch,
                got: msg_kind(&other),
            })
        }
    }
}

fn handle_orchestrator_message(worker_id: usize, epoch: usize, msg: Msg<'_>) -> bool {
    match msg {
        Msg::Control(Command::Disconnect) => {
            info!("disconnect received from orchestrator: worker_id={worker_id} epoch={epoch}");
            true
        }
        other => {
            warn!(
                "unexpected message from orchestrator: worker_id={} epoch={} got={}",
                worker_id,
                epoch,
                msg_kind(&other),
            );
            false
        }
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
