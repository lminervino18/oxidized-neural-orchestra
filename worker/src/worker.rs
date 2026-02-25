use std::{io, net::SocketAddr, num::NonZeroUsize};

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
};
use log::{debug, info, warn};
use machine_learning::training::Trainer;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{error::Result, middleware::Middleware};

/// Infrastructure worker runtime.
pub struct Worker {
    worker_id: usize,
    max_epochs: NonZeroUsize,
    trainer: Box<dyn Trainer>,
}

impl Worker {
    /// Worker runtime that maps weights snapshots into gradient updates.
    ///
    /// # Args
    /// * `worker_id` - Identifier used for observability.
    /// * `max_epochs` - The maximum amount of epochs to run.
    /// * `trainer` - Domain strategy used to compute gradients from weights.
    ///
    /// # Returns
    /// A new worker instance.
    pub fn new(worker_id: usize, max_epochs: NonZeroUsize, trainer: Box<dyn Trainer>) -> Self {
        Self {
            worker_id,
            max_epochs,
            trainer,
        }
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
    pub async fn run<R, W>(
        self,
        mut rx: OnoReceiver<R>,
        mut tx: OnoSender<W>,
        mut middleware: Middleware<R, W>,
    ) -> io::Result<()>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let Self {
            mut trainer,
            max_epochs,
            ..
        } = self;

        let mut rx_buf = vec![0; 1028];
        let mut epoch = 0;

        while epoch < max_epochs.get() {
            tokio::select! {
                param_manager = middlware.pull_params() => {

                }
            }

            let mut param_manager = middleware.pull_params().await?;
            let losses = trainer.train(&mut param_manager);
            middleware.push_grads().await;

            epoch += losses.len();
            // let msg = Msg::Control(Command::ReportLoss { losses });
            // tx.send(&msg).await?;

            match rx.recv_into(&mut rx_buf).await? {
                Msg::Control(Command::Disconnect) => {
                    break;
                }
                msg => {
                    warn!("unexpected message from orchestrator, got: {msg:?}");
                    break;
                }
            }
        }

        Ok(())
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
    ) -> Result<()>
    where
        Rps: AsyncRead + Unpin + Send,
        Wps: AsyncWrite + Unpin + Send,
        Rorch: AsyncRead + Unpin + Send,
        Worch: AsyncWrite + Unpin + Send,
    {
        let worker_id = self.worker_id;
        let mut trainer = self.trainer;
        let mut epoch = 0;

        let mut ps_buf = vec![0; 1028];
        let mut orch_buf = vec![0; 1028];

        while epoch < self.max_epochs.get() {
            debug!("waiting for message");

            tokio::select! {
                ps_msg = ps_rx.recv_into(&mut ps_buf) => {
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

                orch_msg = orch_rx.recv_into(&mut orch_buf) => {
                    let msg = orch_msg?;
                    if handle_orchestrator_message(worker_id, epoch, msg) {
                        break;
                    }
                }
            }
        }

        info!("worker finished");
        let msg = Msg::Control(Command::Disconnect);
        orch_tx.send(&msg).await?;
        ps_tx.send(&msg).await?;

        while !matches!(
            ps_rx.recv_into(&mut ps_buf).await?,
            Msg::Control(Command::Disconnect)
        ) {}

        Ok(())
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
