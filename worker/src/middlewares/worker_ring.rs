use std::{io, num::NonZeroUsize};

use comms::{TransportLayer, WorkerEvent, WorkerHandle};
use machine_learning::param_manager::ParamManager;

use super::SplitIntoChunksMut;

// The communication manager between the worker process
// and both the previous and the next workers.
pub struct WorkerRingManager<T>
where
    T: TransportLayer,
{
    id: usize,
    addrs: Vec<String>,
    prev: WorkerHandle<T>,
    next: WorkerHandle<T>,
    grad: Vec<f32>,
    residual: Vec<f32>,
    amount_of_layers: usize,
}

impl<T> WorkerRingManager<T>
where
    T: TransportLayer,
{
    /// Creates a new `WorkerRingManager`.
    ///
    /// # Args
    /// * `id` - The id of this worker.
    /// * `addrs` - The addresses of every worker in the ring.
    /// * `prev` - The handle for communicating with the previous worker.
    /// * `next` - The handle for communicating with the next worker.
    /// * `size` - The amount of parameters of the model.
    /// * `amount_of_layers` - The amount of layers in the model.
    ///
    /// # Returns
    /// A new `WorkerRingManager` instance.
    pub fn new(
        id: usize,
        addrs: Vec<String>,
        prev: WorkerHandle<T>,
        next: WorkerHandle<T>,
        size: usize,
        amount_of_layers: usize,
    ) -> Self {
        Self {
            id,
            addrs,
            prev,
            next,
            grad: vec![0.0; size],
            residual: vec![0.0; size],
            amount_of_layers,
        }
    }

    /// Creates a new `ParameterManager` binded to this ring manager's buffers.
    ///
    /// # Args
    /// * `params` - The parameters to use for this param manager.
    ///
    /// # Returns
    /// A new `ParameterManager` instance.
    pub fn build_param_manager<'a>(&'a mut self, params: &'a mut [f32]) -> ParamManager<'a> {
        ParamManager::for_worker(
            params,
            &mut self.grad,
            &mut self.residual,
            self.amount_of_layers,
        )
    }

    /// Runs the all reduce algorithm to scatter the partial gradients and
    /// then gather the total aggregated gradient into a `ParamManager`.
    ///
    /// # Returns
    /// A new `ParamManager` instance or an io error if occurred.
    pub async fn pull_grads<'a>(
        &'a mut self,
        params: &'a mut [f32],
    ) -> io::Result<ParamManager<'a>> {
        self.scatter().await?;
        self.gather().await?;
        Ok(self.build_param_manager(params))
    }

    /// Disconnects this worker from the ring of workers.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        self.next.disconnect().await?;

        if let WorkerEvent::Disconnect = self.prev.recv_event().await? {
            return Err(io::Error::other("Expected Disconnect from previous worker"));
        }

        Ok(())
    }

    /// Will start the ring scattering of the gradients with the rest of
    /// the workers.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn scatter(&mut self) -> io::Result<()> {
        let amount_of_workers = self.addrs.len();

        // SAFETY: In the address list there's at list one
        //         address, this worker's address.
        let n = NonZeroUsize::new(amount_of_workers).unwrap();
        let mut chunks: Vec<_> = self.grad.split_chunks_mut(n).collect();
        let mut i = self.id;

        for _ in 0..n.get() - 1 {
            self.next.push_grad(&chunks[i]).await?;
            i = (i + n.get() - 1) % n.get();

            let WorkerEvent::Grad(grad) = self.prev.recv_event().await? else {
                return Err(io::Error::other("Received an invalid worker event"));
            };

            for (acc, g) in chunks[i].into_iter().zip(grad) {
                *acc += g;
            }
        }

        Ok(())
    }

    /// Will start the second step of the all reduce algorithm and gather
    /// all the partial scattered gradients of each worker to aggregate
    /// into a single total gradient.
    ///
    /// # Returns
    /// A new `ParamManager` instance or an io error if occurred.
    async fn gather(&mut self) -> io::Result<()> {
        let amount_of_workers = self.addrs.len();

        // SAFETY: In the address list there's at list one
        //         address, this worker's address.
        let n = NonZeroUsize::new(amount_of_workers).unwrap();
        let mut chunks: Vec<_> = self.grad.split_chunks_mut(n).collect();
        let mut i = (self.id + 1) % n.get();

        for _ in 0..n.get() - 1 {
            self.next.push_grad(&chunks[i]).await?;
            i = (i + n.get() - 1) % n.get();

            let WorkerEvent::Grad(grad) = self.prev.recv_event().await? else {
                return Err(io::Error::other("Received an invalid worker event"));
            };

            for (acc, g) in chunks[i].into_iter().zip(grad) {
                *acc = *g / n.get() as f32;
            }
        }

        Ok(())
    }
}
