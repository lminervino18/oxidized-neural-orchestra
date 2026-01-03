use std::io;

use comms::{
    msg::{Msg, Payload},
    OnoReceiver, OnoSender,
};
use tokio::io::{AsyncRead, AsyncWrite};

/// Parameter Server client wrapper.
///
/// Contract (current protocol):
/// - receive weights as `Msg::Data(Payload::Weights)`
/// - send gradients as `Msg::Data(Payload::Gradient)`
pub struct PsClient<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    rx: OnoReceiver<R>,
    tx: OnoSender<W>,
}

impl<R, W> PsClient<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub fn new(rx: OnoReceiver<R>, tx: OnoSender<W>) -> Self {
        Self { rx, tx }
    }

    /// Receives weights and copies them into `dst`.
    ///
    /// This enforces the "single persistent weights buffer" rule in WorkerState.
    pub async fn recv_weights_into(&mut self, dst: &mut [f32]) -> io::Result<()> {
        let msg: Msg = self.rx.recv().await?;
        match msg {
            Msg::Data(Payload::Weights(w)) => {
                if w.len() != dst.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "weights length mismatch: got {}, expected {}",
                            w.len(),
                            dst.len()
                        ),
                    ));
                }
                dst.copy_from_slice(w);
                Ok(())
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unexpected message: {other:?}"),
            )),
        }
    }

    /// Sends gradients (must match num_params length).
    pub async fn send_grad(&mut self, grad: &[f32]) -> io::Result<()> {
        let msg = Msg::Data(Payload::Gradient(grad));
        self.tx.send(&msg).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io;

    /// Roundtrip test with an in-memory duplex stream:
    /// server sends weights -> worker receives -> worker sends grad -> server receives.
    #[tokio::test]
    async fn test_ps_client_roundtrip_duplex() -> io::Result<()> {
        const PARAMS: usize = 4;
        const BUF_SIZE: usize = 4096;

        let (sv_stream, wk_stream) = io::duplex(BUF_SIZE);

        // Server side
        let (sv_rx, sv_tx) = io::split(sv_stream);
        let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

        // Worker side
        let (wk_rx, wk_tx) = io::split(wk_stream);
        let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);
        let mut client = PsClient::new(wk_rx, wk_tx);

        // 1) Server -> Worker: weights
        let weights = [1.0_f32, 2.0, 3.0, 4.0];
        let msg = Msg::Data(Payload::Weights(&weights));
        sv_tx.send(&msg).await?;

        let mut local = [0.0_f32; PARAMS];
        client.recv_weights_into(&mut local).await?;
        assert_eq!(local, weights);

        // 2) Worker -> Server: grad
        let grad = [0.1_f32, 0.2, 0.3, 0.4];
        client.send_grad(&grad).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Gradient(g)) => assert_eq!(g, grad),
            other => panic!("unexpected msg: {other:?}"),
        }

        Ok(())
    }
}
