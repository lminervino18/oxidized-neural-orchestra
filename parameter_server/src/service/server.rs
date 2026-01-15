use std::io;

use comms::{OnoReceiver, OnoSender};
use tokio::io::{AsyncRead, AsyncWrite};

/// This trait acts as an indirection layer, allowing the `ServerBuilder` to return
/// and manage different `ParameterServer` configurations from it's unique build method.
#[async_trait::async_trait]
pub trait Server<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Indirection method for `ParameterServer::run`.
    async fn run(&mut self) -> io::Result<Vec<f32>>;

    /// Indirection method for `ParameterServer::spawn`
    ///
    /// # Arguments
    /// * `rx` - The receiving end of the communication.
    /// * `tx` - The sending end of the communication.
    fn spawn(&mut self, rx: OnoReceiver<R>, tx: OnoSender<W>);
}
