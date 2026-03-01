use rayon::prelude::*;

use crate::{MlErr, Result, optimization::Optimizer};

/// The state necessary to make forward and backward passes through the network.
pub struct ServerParamsMetadata<'mw> {
    params: &'mw mut [f32],
    grad: &'mw mut [f32],
    acc_grad_buf: &'mw mut [f32],
}

impl<'mw> ServerParamsMetadata<'mw> {
    /// Creates a new `ServerParamsMetadata`.
    ///
    /// # Arguments
    /// * `params` - The mutable slice of this server's parameters.
    /// * `grad` - The server's dedicated gradient slice.
    /// * `acc_grad_buf` - The server's accumulated gradient buffer.
    ///
    /// # Returns
    /// A new `ServerParamsMetadata` instance.
    pub fn new(params: &'mw mut [f32], grad: &'mw mut [f32], acc_grad_buf: &'mw mut [f32]) -> Self {
        Self {
            params,
            grad,
            acc_grad_buf,
        }
    }
}

/// The manager of parameters, this middleware's module manages the model's parameter retrieval from the
/// servers and selects which set of parameters to use for each layer of the model's trining when traversing
/// it's layers forwards and backwards.
pub struct ParamManager<'mw> {
    servers: Vec<ServerParamsMetadata<'mw>>,
    server_ordering: &'mw [usize],
    cursors: Vec<usize>,
}

impl<'mw> ParamManager<'mw> {
    /// Creates a new `ParamManager`.
    ///
    /// # Arguments
    /// * `servers` - A list of the necessary server metadata to have to iterate through the layers' parameters.
    /// * `server_ordering` - The ordering of the servers to know which layer corresponds to which server.
    ///
    /// # Returns
    /// A new `ParamManager` instance.
    pub fn new(servers: Vec<ServerParamsMetadata<'mw>>, server_ordering: &'mw [usize]) -> Self {
        Self {
            cursors: vec![0; server_ordering.len()],
            server_ordering,
            servers,
        }
    }

    /// Creates a new `FrontIter` parameter iterator.
    ///
    /// The returned iterator iterates the model's layers forward.
    ///
    /// # Returns
    /// A new `FrontIter` instance.
    pub fn front<'pm>(&'pm mut self) -> FrontIter<'pm, 'mw> {
        self.cursors.fill(0);

        FrontIter {
            servers: &mut self.servers,
            server_ordering: self.server_ordering,
            cursors: &mut self.cursors,
            curr: 0,
        }
    }

    /// Creates a new `BackIter` parameter iterator.
    ///
    /// The returned iterator iterates the model's layers backwards.
    ///
    /// # Returns
    /// A new `Backiter` instance.
    pub fn back<'pm>(&'pm mut self) -> BackIter<'pm, 'mw> {
        self.cursors.fill(0);

        BackIter {
            servers: &mut self.servers,
            server_ordering: &self.server_ordering,
            cursors: &mut self.cursors,
            curr: 0,
        }
    }

    /// Applies the gradients onto the parameters of the model.
    ///
    /// # Arguments
    /// * `optimizers` - A list of optimizers, one per server.
    pub fn optimize<O: Optimizer + Send>(&mut self, optimizers: &mut [O]) -> Result<()> {
        if optimizers.len() != self.servers.len() {
            return Err(MlErr::SizeMismatch {
                what: "optimizers",
                got: optimizers.len(),
                expected: self.servers.len(),
            });
        }

        optimizers
            .par_iter_mut()
            .zip(&mut self.servers)
            .try_for_each(|(optimizer, server)| {
                optimizer.update_params(&server.grad, server.params)
            })?;

        Ok(())
    }

    /// Zeroes out the gradients of every server.
    pub fn zero_grad(&mut self) {
        self.servers
            .par_iter_mut()
            .for_each(|server| server.grad.fill(0.0));
    }

    /// Accumulates the current gradient onto the inner accumulated gradients buffer.
    pub fn acc_grad(&mut self) {
        self.servers.par_iter_mut().for_each(|server| {
            for (acc, g) in server.acc_grad_buf.iter_mut().zip(server.grad.iter()) {
                *acc += *g;
            }
        });
    }
}

/// A model's layer iterator.
///
/// This iterator iterates the layers of a model from the front.
pub struct FrontIter<'pm, 'mw> {
    servers: &'pm mut [ServerParamsMetadata<'mw>],
    server_ordering: &'mw [usize],
    cursors: &'pm mut [usize],
    curr: usize,
}

impl FrontIter<'_, '_> {
    /// Tries to yield the next layer's parameters and gradient.
    ///
    /// # Arguments
    /// * `n` - The amount of parameters to take from the inner storage of the next server.
    ///
    /// # Returns
    /// An option denoting if there still are more parameters and gradients.
    pub fn next(&mut self, n: usize) -> Option<&mut [f32]> {
        if self.curr == self.server_ordering.len() {
            return None;
        }

        let server_id = self.server_ordering[self.curr];
        let server = &mut self.servers[server_id];
        let start = self.cursors[server_id];
        let end = (start + n).min(server.params.len());

        self.cursors[server_id] = end;
        self.curr += 1;

        Some(&mut server.params[start..end])
    }
}

/// A model's layer iterator.
///
/// This iterator iterates the layers of a model from the back.
pub struct BackIter<'pm, 'mw> {
    servers: &'pm mut [ServerParamsMetadata<'mw>],
    server_ordering: &'mw [usize],
    cursors: &'pm mut [usize],
    curr: usize,
}

impl BackIter<'_, '_> {
    /// Tries to yield the next layer's parameters and gradient.
    ///
    /// # Arguments
    /// * `n` - The amount of parameters to take from the inner storage of the next server.
    ///
    /// # Returns
    /// An option denoting if there still are more parameters and gradients.
    pub fn next(&mut self, n: usize) -> Option<(&mut [f32], &mut [f32])> {
        if self.curr == self.server_ordering.len() {
            return None;
        }

        let idx = self.server_ordering.len() - self.curr - 1;
        let server_id = self.server_ordering[idx];
        let server = &mut self.servers[server_id];
        let end = server.params.len() - self.cursors[server_id];
        let start = end.saturating_sub(n);

        self.cursors[server_id] += end - start;
        self.curr += 1;

        Some((&mut server.params[start..end], &mut server.grad[start..end]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_params_grads(server_sizes: &[usize]) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        server_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| (vec![i as f32; size], vec![i as f32; size], vec![0.0; size]))
            .collect()
    }

    #[test]
    fn front() {
        const NSERVERS: usize = 2;
        const SERVER_SIZES: [usize; NSERVERS] = [19, 20];
        const NLAYERS: usize = 4;
        const LAYER_SIZES: [usize; NLAYERS] = [9, 15, 5, 10];
        const ORDERING: [usize; NLAYERS] = [0, 1, 1, 0];

        let mut params_grads = gen_params_grads(&SERVER_SIZES);
        let servers: Vec<_> = params_grads
            .iter_mut()
            .map(|(params, grad, acc_grad_buf)| {
                ServerParamsMetadata::new(params, grad, acc_grad_buf)
            })
            .collect();

        let mut manager = ParamManager::new(servers, &ORDERING);
        let mut front = manager.front();

        for (i, size) in (0..LAYER_SIZES.len()).zip(LAYER_SIZES) {
            let params = front.next(size).unwrap();
            assert_eq!(params[0], ORDERING[i] as f32);
            assert_eq!(params.len(), LAYER_SIZES[i]);
        }
    }

    #[test]
    fn back() {
        const NSERVERS: usize = 3;
        const SERVER_SIZES: [usize; NSERVERS] = [19, 20, 5];
        const NLAYERS: usize = 5;
        const LAYER_SIZES: [usize; NLAYERS] = [9, 15, 5, 5, 10];
        const ORDERING: [usize; NLAYERS] = [0, 1, 1, 2, 0];

        let mut params_grads = gen_params_grads(&SERVER_SIZES);
        let servers: Vec<_> = params_grads
            .iter_mut()
            .map(|(params, grad, acc_grad_buf)| {
                ServerParamsMetadata::new(params, grad, acc_grad_buf)
            })
            .collect();

        let mut manager = ParamManager::new(servers, &ORDERING);
        let mut back = manager.back();

        for (i, size) in (0..LAYER_SIZES.len()).zip(LAYER_SIZES).rev() {
            let (params, grad) = back.next(size).unwrap();
            assert_eq!(params[0], ORDERING[i] as f32);
            assert_eq!(params.len(), LAYER_SIZES[i]);

            assert_eq!(grad[0], ORDERING[i] as f32);
            assert_eq!(grad.len(), LAYER_SIZES[i]);
        }
    }

    #[test]
    fn front_back() {
        const NSERVERS: usize = 4;
        const SERVER_SIZES: [usize; NSERVERS] = [19, 20, 21, 20];
        const NLAYERS: usize = 8;
        const LAYER_SIZES: [usize; NLAYERS] = [9, 15, 5, 15, 1, 20, 5, 10];
        const ORDERING: [usize; NLAYERS] = [0, 1, 2, 2, 2, 3, 1, 0];

        let mut params_grads = gen_params_grads(&SERVER_SIZES);
        let servers: Vec<_> = params_grads
            .iter_mut()
            .map(|(params, grad, acc_grad_buf)| {
                ServerParamsMetadata::new(params, grad, acc_grad_buf)
            })
            .collect();

        let mut manager = ParamManager::new(servers, &ORDERING);
        let mut back = manager.back();

        for (i, size) in (0..LAYER_SIZES.len()).zip(LAYER_SIZES).rev() {
            let (params, grad) = back.next(size).unwrap();
            assert_eq!(params[0], ORDERING[i] as f32);
            assert_eq!(params.len(), LAYER_SIZES[i]);

            assert_eq!(grad[0], ORDERING[i] as f32);
            assert_eq!(grad.len(), LAYER_SIZES[i]);
        }
    }
}
