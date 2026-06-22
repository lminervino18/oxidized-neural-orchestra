use std::io;

use futures::future;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{ParamServerHandle, TransportLayer};

/// A helper struct to manage a cluster of parameter servers.
pub struct ParamServerCluster<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    server_handles: Vec<ParamServerHandle<R, W, T>>,
}

impl<R, W, T> Default for ParamServerCluster<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    fn default() -> Self {
        Self {
            server_handles: Default::default(),
        }
    }
}

impl<R, W, T> ParamServerCluster<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    /// Creates a new `ParamServerCluster`.
    ///
    /// # Returns
    /// A new `ParamServerCluster` instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new server communicator to the cluster.
    ///
    /// # Args
    /// * `server_handle` - The handler for communicating with a new server.
    pub fn spawn(&mut self, server_handle: ParamServerHandle<R, W, T>) {
        self.server_handles.push(server_handle);
    }

    /// Pulls the new parameters from all the servers.
    ///
    /// # Returns
    /// The parameters from all the servers or io errors if occurred.
    pub async fn pull_params(&mut self) -> Vec<io::Result<&mut [f32]>> {
        let futs = self
            .server_handles
            .iter_mut()
            .map(async |server_handle| server_handle.pull_params().await);

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

    /// Waits till receiving a message and discards it.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn discard_one(&mut self) -> io::Result<()> {
        let futs = self
            .server_handles
            .iter_mut()
            .map(async |server_handle| server_handle.discard_one().await);

        future::join_all(futs).await;
        Ok(())
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
