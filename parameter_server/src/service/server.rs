use std::io;

use comms::{TransportLayer, WorkerHandle};

/// This trait acts as an indirection layer, allowing the `ServerBuilder` to return
/// and manage different `ParameterServer` configurations from it's unique build method.
#[async_trait::async_trait]
pub trait Server<T>: Send
where
    T: TransportLayer,
{
    /// Indirection method for `ParameterServer::run`.
    async fn run(&mut self) -> io::Result<Vec<f32>>;

    /// Indirection method for `ParameterServer::spawn`
    ///
    /// # Args
    /// * `worker_handle` - The handle to enable communication with the worker.
    fn spawn(&mut self, worker_handle: WorkerHandle<T>);
}
