use std::io;

use comms::{ParamServerCluster, ParamServerHandle, TransportLayer};
use machine_learning::param_manager::{ParamManager, ServerParamsMetadata};

// The communication manager between the worker process and the many servers.
pub struct ServerClusterManager<T>
where
    T: TransportLayer,
{
    cluster: ParamServerCluster<T>,
    server_ordering: Vec<usize>,
    residuals: Vec<Vec<f32>>,
    grads: Vec<Vec<f32>>,
}

impl<T> ServerClusterManager<T>
where
    T: TransportLayer,
{
    /// Creates a new `ServerClusterManager`.
    ///
    /// # Args
    /// * `server_ordering`: The ordering of the servers to know which layer's parameters correspond to which server.
    ///
    /// # Returns
    /// A new `ServerClusterManager` instance.
    pub fn new(server_ordering: Vec<usize>) -> Self {
        Self {
            cluster: ParamServerCluster::new(),
            server_ordering,
            residuals: Vec::new(),
            grads: Vec::new(),
        }
    }

    /// Adds a new server communicator to the middleware.
    ///
    /// # Args
    /// * `server_handle` - The handle to the parameter server.
    /// * `size` - The amount of parameters this server holds.
    pub fn spawn(&mut self, server_handle: ParamServerHandle<T>, size: usize) {
        self.cluster.spawn(server_handle);
        self.residuals.push(vec![0.0; size]);
        self.grads.push(vec![0.0; size]);
    }

    /// Pulls the new parameters from all the servers.
    ///
    /// # Returns
    /// A new `ParamManager` instance with all the parameters.
    pub async fn pull_params(&mut self) -> io::Result<ParamManager<'_>> {
        let cluster_params = self.cluster.pull_params().await;
        let mut metadatas = Vec::with_capacity(cluster_params.len());

        for ((res, grad), residual) in cluster_params
            .into_iter()
            .zip(&mut self.grads)
            .zip(&mut self.residuals)
        {
            match res {
                Ok(params) => {
                    let metadata = ServerParamsMetadata::new(params, grad, residual);
                    metadatas.push(metadata);
                }
                Err(_) => {
                    todo!("detected a server disconnect");
                }
            }
        }

        Ok(ParamManager::new(metadatas, &self.server_ordering))
    }

    /// Pushes the latest gradients to the servers.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_grads(&mut self) -> io::Result<()> {
        let thresholds = self.cluster.push_grads(&self.residuals).await;

        for (residual, threshold) in self.residuals.iter_mut().zip(thresholds) {
            match threshold {
                Ok(None) => residual.fill(0.0),
                Ok(Some(t)) => {
                    residual
                        .iter_mut()
                        .filter(|g| g.abs() >= t)
                        .for_each(|g| *g = 0.0);
                }
                _ => {
                    todo!("detected a server disconnection");
                }
            }
        }

        Ok(())
    }

    /// Disconnects this worker from all the servers.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        self.cluster.disconnect().await
    }
}
