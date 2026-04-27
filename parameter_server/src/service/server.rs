use std::io;

use comms::{Rtp, WorkerHandle};
use tokio::io::{AsyncRead, AsyncWrite};

/// This trait acts as an indirection layer, allowing the `ServerBuilder` to return
/// and manage different `ParameterServer` configurations from it's unique build method.
#[async_trait::async_trait]
pub trait Server<R, W>: Send
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    /// Indirection method for `ParameterServer::run`.
    async fn run(&mut self) -> io::Result<Vec<f32>>;

    /// Indirection method for `ParameterServer::spawn`
    ///
    /// # Args
    /// * `worker_handle` - The handle to enable communication with the worker.
    fn spawn(&mut self, worker_handle: WorkerHandle<Rtp<R, W>>);
}
