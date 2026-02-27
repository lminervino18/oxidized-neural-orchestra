use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg},
};
use log::{debug, info, warn};
use machine_learning::training::{TrainResult, Trainer};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::middleware::Middleware;

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
        let mut trainer = self.trainer;
        let mut rx_buf = vec![0; 1028];
        let mut should_continue = true;

        while should_continue {
            tokio::select! {
                ret = middleware.pull_params() => {
                    debug!("received parameters from all servers, training...");

                    let mut param_manager = ret?;

                    // TODO: Handlear esto mejor, capaz podemos mandarle mensajes a todos los servidores
                    //       para que reintenten de enviar el ultimo mensaje o algo asi (todavia no se
                    //       si tiene sentido tampoco que les avisemos, despues de todo ellos mandaron
                    //       mal el mensaje)
                    let TrainResult { losses, was_last } = trainer.train(&mut param_manager).unwrap();
                    middleware.push_grads().await?;

                    should_continue = !was_last;
                    let msg = Msg::Control(Command::ReportLoss { losses });
                    tx.send(&msg).await?;
                }
                ret = rx.recv_into(&mut rx_buf) => match ret? {
                    Msg::Control(Command::Disconnect) => {
                        info!("received a Command::Disconnect from the orchestrator");
                        break;
                    }
                    other => {
                        warn!("unexpected message from orchestrator, got: {other:?}");
                    }
                }
            }
        }

        middleware.disconnect().await?;
        let msg = Msg::Control(Command::Disconnect);
        tx.send(&msg).await?;
        Ok(())
    }
}
