// TODO: arreglar los docstrings

use rayon::prelude::*;

use super::ServerParamsMetadata;
use crate::optimization::Optimizer;

/// The manager of parameters, this middleware's module manages the model's parameter retrieval from the
/// servers and selects which set of parameters to use for each layer of the model's trining when traversing
/// it's layers forwards and backwards.
pub struct ParamManager<'mw> {
    servers: Vec<ServerParamsMetadata<'mw>>,
    server_ordering: &'mw [usize],
    layer_sizes: &'mw [usize],
    cursors: Vec<usize>,
}

impl<'mw> ParamManager<'mw> {
    /// Creates a new `ParamManager`.
    ///
    /// # Arguments
    /// * `middleware` - The middleware for the communication between the worker and the server.
    ///
    /// # Returns
    /// A new `ParamManager` instance.
    pub fn new(
        servers: Vec<ServerParamsMetadata<'mw>>,
        server_ordering: &'mw [usize],
        layer_sizes: &'mw [usize],
    ) -> Self {
        Self {
            cursors: vec![0; layer_sizes.len()],
            server_ordering,
            layer_sizes,
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
            layer_sizes: self.layer_sizes,
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
            layer_sizes: &self.layer_sizes,
            cursors: &mut self.cursors,
            curr: 0,
        }
    }

    /// Applies the gradients onto the parameters of the model.
    ///
    /// # Arguments
    /// * `optimizers` - A list of optimizers, one per server.
    pub fn optimize<O: Optimizer>(&mut self, optimizers: &mut [O]) {
        // TODO: Agregar un chequeo para validar que el largo de los servers y optimizers sea el mismo
        optimizers
            .par_iter_mut()
            .zip(&mut self.servers)
            .for_each(|(optimizer, server)| {
                optimizer.update_params(server.params, server.grad);
            });
    }

    /// Zeros out the gradients of every server.
    pub fn zero_grad(&mut self) {
        self.servers
            .par_iter_mut()
            .for_each(|server| server.grad.fill(0.0));
    }
}

/// A model's layer iterator.
///
/// This iterator iterates the layers of a model from the front.
pub struct FrontIter<'pm, 'mw> {
    servers: &'pm mut [ServerParamsMetadata<'mw>],
    server_ordering: &'mw [usize],
    layer_sizes: &'mw [usize],
    cursors: &'pm mut [usize],
    curr: usize,
}

impl FrontIter<'_, '_> {
    /// Tries to yield the next layer's parameters and gradient.
    ///
    /// # Returns
    /// An option denoting if there still are more parameters and gradients.
    pub fn next(&mut self) -> Option<&mut [f32]> {
        if self.curr == self.server_ordering.len() {
            return None;
        }

        let server_id = self.server_ordering[self.curr];
        let layer_size = self.layer_sizes[self.curr];
        let server = &mut self.servers[server_id];
        let start = self.cursors[server_id];
        let end = start + layer_size;

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
    layer_sizes: &'mw [usize],
    cursors: &'pm mut [usize],
    curr: usize,
}

impl BackIter<'_, '_> {
    /// Tries to yield the next layer's parameters and gradient.
    ///
    /// # Returns
    /// An option denoting if there still are more parameters and gradients.
    pub fn next(&mut self) -> Option<(&mut [f32], &mut [f32])> {
        if self.curr == self.server_ordering.len() {
            return None;
        }

        let idx = self.server_ordering.len() - self.curr - 1;
        let server_id = self.server_ordering[idx];
        let layer_size = self.layer_sizes[idx];
        let server = &mut self.servers[server_id];
        let start = self.cursors[server_id];
        let end = start + layer_size;

        self.cursors[server_id] = end;
        self.curr += 1;

        Some((&mut server.params[start..end], &mut server.grad[start..end]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_params_grads(server_sizes: &[usize]) -> Vec<(Vec<f32>, Vec<f32>)> {
        server_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| (vec![i as f32; size], vec![i as f32; size]))
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
            .map(|(params, grad)| ServerParamsMetadata { params, grad })
            .collect();

        let mut manager = ParamManager::new(servers, &ORDERING, &LAYER_SIZES);
        let mut front = manager.front();

        for i in 0..LAYER_SIZES.len() {
            let params = front.next().unwrap();
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
            .map(|(params, grad)| ServerParamsMetadata { params, grad })
            .collect();

        let mut manager = ParamManager::new(servers, &ORDERING, &LAYER_SIZES);
        let mut back = manager.back();

        for i in (0..LAYER_SIZES.len()).rev() {
            let (params, grad) = back.next().unwrap();
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
            .map(|(params, grad)| ServerParamsMetadata { params, grad })
            .collect();

        let mut manager = ParamManager::new(servers, &ORDERING, &LAYER_SIZES);
        let mut back = manager.back();

        for i in (0..LAYER_SIZES.len()).rev() {
            let (params, grad) = back.next().unwrap();
            assert_eq!(params[0], ORDERING[i] as f32);
            assert_eq!(params.len(), LAYER_SIZES[i]);

            assert_eq!(grad[0], ORDERING[i] as f32);
            assert_eq!(grad.len(), LAYER_SIZES[i]);
        }
    }
}
