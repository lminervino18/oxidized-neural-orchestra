use std::io;

use futures::future;

use crate::{ParamServerHandle, PullParamsResponse, TransportLayer};

/// A helper struct to manage a cluster of parameter servers.
pub struct ParamServerCluster<T>
where
    T: TransportLayer,
{
    server_handles: Vec<ParamServerHandle<T>>,
}

impl<T> ParamServerCluster<T>
where
    T: TransportLayer,
{
    /// Creates a new `ParamServerCluster`.
    ///
    /// # Returns
    /// A new `ParamServerCluster` instance.
    pub fn new() -> Self {
        Self {
            server_handles: Vec::new(),
        }
    }

    /// Adds a new server communicator to the cluster.
    ///
    /// # Args
    /// * `server_handle` - The handler for communicating with a new server.
    pub fn spawn(&mut self, server_handle: ParamServerHandle<T>) {
        self.server_handles.push(server_handle);
    }

    /// Pulls the new parameters from all the servers.
    ///
    /// # Returns
    /// The parameters from all the servers or io errors if occurred.
    pub async fn pull_params(&mut self) -> Vec<io::Result<&mut [f32]>> {
        let futs = self.server_handles.iter_mut().map(async |server_handle| {
            match server_handle.pull_params().await? {
                PullParamsResponse::Params(params) => Ok::<_, io::Error>(params),
            }
        });

        future::join_all(futs).await
    }

    /// Pushes the latest gradients to the servers.
    ///
    /// # Returns
    /// The thresholds for cleaning the residual vecs or io errors if occurred.
    pub async fn push_grads(&mut self, residuals: &[Vec<f32>]) -> Vec<io::Result<Option<f32>>> {
        let futs = self
            .server_handles
            .iter_mut()
            .zip(residuals)
            .map(async |(server_handle, residual)| server_handle.push_grad(residual).await);

        future::join_all(futs).await
    }

    /// Disconnects this worker from the cluster.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let futs = self
            .server_handles
            .iter_mut()
            .map(async |server_handle| server_handle.disconnect().await);

        future::try_join_all(futs).await?;
        Ok(())
    }
}
