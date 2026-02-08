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
    algorithm: AlgorithmSpec,
    trainer: Box<dyn Trainer>,
}

impl Worker {
    /// Worker runtime that maps weights snapshots into gradient updates.
    ///
    /// # Args
    /// * `worker_id` - Identifier used for observability.
    /// * `algorithm` - Distributed algorithm selection for this worker.
    /// * `trainer` - Domain strategy used to compute gradients from weights.
    ///
    /// # Returns
    /// A new worker instance.
    pub fn new(worker_id: usize, algorithm: AlgorithmSpec, trainer: Box<dyn Trainer>) -> Self {
        Self {
            worker_id,
            algorithm,
            trainer,
        }
    }

    /// Runs the worker using its configured distributed algorithm.
    ///
    /// # Returns
    /// Returns `Ok(())` on graceful completion.
    ///
    /// # Errors
    /// Returns `WorkerError` on I/O failures or protocol violations.
    pub async fn run(self) -> Result<(), WorkerError> {
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

                self.run_parameter_server(ps_rx, ps_tx).await
            }
        }
    }

    /// Runs the parameter-server protocol loop until `Disconnect` or a violation.
    ///
    /// # Args
    /// * `rx` - Receiving channel end.
    /// * `tx` - Sending channel end.
    ///
    /// # Errors
    /// Returns `WorkerError` on I/O failures or protocol invariant violations.
    pub async fn run_parameter_server<R, W>(
        self,
        mut rx: OnoReceiver<R>,
        mut tx: OnoSender<W>,
    ) -> Result<(), WorkerError>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        let worker_id = self.worker_id;
        let mut trainer = self.trainer;

        let mut step: usize = 0;

        info!("worker starting: worker_id={worker_id}");

        loop {
            debug!("waiting message: worker_id={worker_id} step={step}");

            let msg: Msg = rx.recv().await.map_err(WorkerError::Io)?;

            match msg {
                Msg::Control(Command::Disconnect) => {
                    info!("disconnect received: worker_id={worker_id} step={step}");
                    break;
                }

                Msg::Data(Payload::Params(weights)) => {
                    let got = weights.len();
                    debug!("training: worker_id={worker_id} step={step} params={got}");

                    let grad = trainer.train(weights);

                    tx.send(&Msg::Data(Payload::Grad(grad)))
                        .await
                        .map_err(WorkerError::Io)?;

                    step += 1;
                }

                other => {
                    warn!(
                        "unexpected message: worker_id={} step={} got={}",
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

        info!("worker finished: worker_id={worker_id} steps={step}");
        Ok(())
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
