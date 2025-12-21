use std::sync::atomic::Ordering;

use rayon::prelude::*;

use crate::{
    optimization::Optimizer,
    parameter_server::{PSClient, SharedData},
};

/// Parameter Server implementation that handles model weights and gradients aggregation.
///
/// `PSServer` coordinates the transition between gradient accumulation (performed by worker threads) and weight updates
/// (peformed by the main server thread). It uses a double-buffering strategy to allow worker threads to continue
/// accumulating gradients into one buffer while the server thread processes the other.
///
/// # Type Parameters
/// * `O` - An implementation of the `Optimizer` trait used to update weights.
#[derive(Debug)]
pub struct PSServer<O>
where
    O: Optimizer,
{
    /// Shared handle to the atomic gradient buffers and synchronization primitives.
    client: PSClient,
    /// Intermediate flat buffer used to store a snapshot of gradients for the optimizer.
    grad_buf: Vec<f32>,
    /// The current model parameters being optimized.
    weights: Vec<f32>,
    /// The optimization algorithm.
    optimizer: O,
}

impl<O> PSServer<O>
where
    O: Optimizer,
{
    /// Creates a new `PSServer` instance.
    ///
    /// # Arguments
    /// * `n` - The number of parameters (weights) in the model.
    /// * `optimizer` - The strategy used to update weights from accumulated gradients.
    pub fn new(n: usize, optimizer: O) -> Self {
        Self {
            client: PSClient::new(SharedData::new(n)),
            grad_buf: vec![0.; n],
            weights: vec![0.; n],
            optimizer,
        }
    }

    /// Returns a thread-safe handle `PSClient` that can be distributed to worker threads.
    ///
    /// Workers use this handle to call `accumulate` without needing exclusive access to the server.
    pub fn client_handle(&self) -> PSClient {
        self.client.clone()
    }

    /// Performs a weight update based on all gradients accumulated since the last call.
    ///
    /// Will swap the underlying gradient and read it, clearing it in the process, later will call the optimizer on
    /// this gradient and weights.
    pub fn update_weights(&mut self) {
        let grad = self.client.swap_grad();

        self.grad_buf
            .par_iter_mut()
            .zip(grad.par_iter())
            .for_each(|(dst, src)| {
                *dst = src.swap(0., Ordering::Relaxed);
            });

        self.optimizer
            .update_weights(&mut self.weights, &self.grad_buf);
    }
}
