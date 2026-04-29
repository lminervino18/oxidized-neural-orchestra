use rayon::prelude::*;

use crate::{MlErr, Result, optimization::Optimizer};

/// The different variants of ordering the layers.
/// Either have different ids per layer or a constant owner for all of them.
#[derive(Clone, Copy)]
enum LayerOrdering<'a> {
    Seq(&'a [usize]),
    Const(usize, usize),
}

impl<'a> LayerOrdering<'a> {
    fn len(&self) -> usize {
        match self {
            Self::Seq(items) => items.len(),
            Self::Const(.., n) => *n,
        }
    }

    fn nth(&self, i: usize) -> Option<usize> {
        match self {
            Self::Seq(items) => items.get(i).copied(),
            Self::Const(x, n) if i < *n => Some(*x),
            Self::Const(..) => None,
        }
    }
}

/// The state necessary to make forward and backward passes through the network.
pub struct ParamsMetadata<'mw> {
    params: &'mw mut [f32],
    grad: &'mw mut [f32],
    residual: &'mw mut [f32],
}

impl<'mw> ParamsMetadata<'mw> {
    /// Creates a new `ServerParamsMetadata`.
    ///
    /// # Args
    /// * `params` - The mutable slice of this server's parameters.
    /// * `grad` - The server's dedicated gradient slice.
    /// * `residual` - The server's accumulated gradient buffer.
    ///
    /// # Returns
    /// A new `ServerParamsMetadata` instance.
    pub fn new(params: &'mw mut [f32], grad: &'mw mut [f32], residual: &'mw mut [f32]) -> Self {
        Self {
            params,
            grad,
            residual,
        }
    }
}

/// The manager of parameters, this middleware's module manages the model's parameter retrieval from the
/// servers and selects which set of parameters to use for each layer of the model's trining when traversing
/// it's layers forwards and backwards.
pub struct ParamManager<'mw> {
    metadatas: Vec<ParamsMetadata<'mw>>,
    layer_ordering: LayerOrdering<'mw>,
    cursors: Vec<usize>,
}

impl<'mw> ParamManager<'mw> {
    /// Creates a new `ParamManager`.
    ///
    /// # Args
    /// * `servers` - A list of the necessary server metadata to have to iterate through the layers' parameters.
    /// * `layer_ordering` - The ordering of the layers to know which entity holds which layer.
    ///
    /// # Returns
    /// A new `ParamManager` instance.
    pub fn for_servers(servers: Vec<ParamsMetadata<'mw>>, layer_ordering: &'mw [usize]) -> Self {
        Self {
            cursors: vec![0; layer_ordering.len()],
            layer_ordering: LayerOrdering::Seq(layer_ordering),
            metadatas: servers,
        }
    }

    /// Creates a new `ParamManager`.
    ///
    /// # Args
    /// * `params` - The parameters of the worker.
    /// * `grad` - The gradient buffer of the worker.
    /// * `residual` - The accumulative gradient buffer of the worker.
    /// * `amount_of_layers` - The amount of layers in the model.
    ///
    /// # Returns
    /// A new `ParamManager` instance.
    pub fn for_worker(
        params: &'mw mut [f32],
        grad: &'mw mut [f32],
        residual: &'mw mut [f32],
        amount_of_layers: usize,
    ) -> Self {
        Self {
            cursors: vec![0],
            layer_ordering: LayerOrdering::Const(0, amount_of_layers),
            metadatas: vec![ParamsMetadata::new(params, grad, residual)],
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
            metadatas: &mut self.metadatas,
            layer_ordering: self.layer_ordering,
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
            metadatas: &mut self.metadatas,
            layer_ordering: self.layer_ordering,
            cursors: &mut self.cursors,
            curr: 0,
        }
    }

    /// Applies the gradients onto the parameters of the model.
    ///
    /// # Args
    /// * `optimizers` - A list of optimizers, one per entity.
    pub fn optimize<O: Optimizer + Send>(&mut self, optimizers: &mut [O]) -> Result<()> {
        if optimizers.len() != self.metadatas.len() {
            return Err(MlErr::SizeMismatch {
                what: "optimizers",
                got: optimizers.len(),
                expected: self.metadatas.len(),
            });
        }

        optimizers
            .par_iter_mut()
            .zip(&mut self.metadatas)
            .try_for_each(|(optimizer, metadata)| {
                optimizer.update_params(metadata.grad, metadata.params)
            })?;

        Ok(())
    }

    /// Zeroes out the gradients of every entity.
    pub fn zero_grad(&mut self) {
        self.metadatas
            .par_iter_mut()
            .for_each(|metadata| metadata.grad.fill(0.0));
    }

    /// Accumulates the current gradient onto the inner accumulated residual buffer.
    pub fn acc_residual(&mut self) {
        self.metadatas.par_iter_mut().for_each(|metadata| {
            for (acc, g) in metadata.residual.iter_mut().zip(metadata.grad.iter()) {
                *acc += *g;
            }
        });
    }
}

/// A model's layer iterator.
///
/// This iterator iterates the layers of a model from the front.
pub struct FrontIter<'pm, 'mw> {
    metadatas: &'pm mut [ParamsMetadata<'mw>],
    layer_ordering: LayerOrdering<'mw>,
    cursors: &'pm mut [usize],
    curr: usize,
}

impl FrontIter<'_, '_> {
    /// Tries to yield the next layer's parameters and gradient.
    ///
    /// If the requested size is `0`, then the iterator will yield two empty
    /// buffers without advancing it's internal ordering pointer. This becomes
    /// handy for when trying to request parameters for a stateless (in terms
    /// of parameters) layer.
    ///
    /// # Args
    /// * `n` - The amount of parameters to take from the inner storage of the next entity.
    ///
    /// # Returns
    /// An option denoting if there still are more parameters and gradients.
    pub fn next(&mut self, n: usize) -> Option<&mut [f32]> {
        if n == 0 {
            return Some(&mut []);
        }

        if self.curr >= self.layer_ordering.len() {
            return None;
        }

        // SAFETY: curr must be smaller than layer_ordering's length.
        let id = self.layer_ordering.nth(self.curr).unwrap();
        let metadata = &mut self.metadatas[id];
        let start = self.cursors[id];
        let end = (start + n).min(metadata.params.len());

        self.cursors[id] = end;
        self.curr += 1;

        Some(&mut metadata.params[start..end])
    }
}

/// A model's layer iterator.
///
/// This iterator iterates the layers of a model from the back.
pub struct BackIter<'pm, 'mw> {
    metadatas: &'pm mut [ParamsMetadata<'mw>],
    layer_ordering: LayerOrdering<'mw>,
    cursors: &'pm mut [usize],
    curr: usize,
}

impl BackIter<'_, '_> {
    /// Tries to yield the next layer's parameters and gradient.
    ///
    /// If the requested size is `0`, then the iterator will yield two empty
    /// buffers without advancing it's internal ordering pointer. This becomes
    /// handy for when trying to request parameters for a stateless (in terms
    /// of parameters) layer.
    ///
    /// # Args
    /// * `n` - The amount of parameters to take from the inner storage of the next entity.
    ///
    /// # Returns
    /// An option denoting if there still are more parameters and gradients.
    pub fn next(&mut self, n: usize) -> Option<(&mut [f32], &mut [f32])> {
        if n == 0 {
            return Some((&mut [], &mut []));
        }

        if self.curr >= self.layer_ordering.len() {
            return None;
        }

        let idx = self.layer_ordering.len() - self.curr - 1;

        // SAFETY: idx must be inside the valid range.
        let id = self.layer_ordering.nth(idx).unwrap();
        let metadata = &mut self.metadatas[id];
        let end = metadata.params.len() - self.cursors[id];
        let start = end.saturating_sub(n);

        self.cursors[id] += end - start;
        self.curr += 1;

        Some((
            &mut metadata.params[start..end],
            &mut metadata.grad[start..end],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_params_grads(sizes: &[usize]) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        sizes
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
            .map(|(params, grad, residual)| ParamsMetadata::new(params, grad, residual))
            .collect();

        let mut manager = ParamManager::for_servers(servers, &ORDERING);
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
            .map(|(params, grad, residual)| ParamsMetadata::new(params, grad, residual))
            .collect();

        let mut manager = ParamManager::for_servers(servers, &ORDERING);
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
            .map(|(params, grad, residual)| ParamsMetadata::new(params, grad, residual))
            .collect();

        let mut manager = ParamManager::for_servers(servers, &ORDERING);
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
    fn zero_size() {
        const NSERVERS: usize = 2;
        const SERVER_SIZES: [usize; NSERVERS] = [1, 2];
        const NLAYERS: usize = 3;
        const ORDERING: [usize; NLAYERS] = [1, 0, 1];

        let mut params_grads = gen_params_grads(&SERVER_SIZES);
        let servers: Vec<_> = params_grads
            .iter_mut()
            .map(|(params, grad, residual)| ParamsMetadata::new(params, grad, residual))
            .collect();

        let mut manager = ParamManager::for_servers(servers, &ORDERING);
        let mut front = manager.front();

        assert_eq!(front.next(1).unwrap(), &[1.0]);
        assert!(front.next(0).unwrap().is_empty());
        assert_eq!(front.next(1).unwrap(), &[0.0]);
        assert!(front.next(0).unwrap().is_empty());
        assert_eq!(front.next(1).unwrap(), &[1.0]);
    }
}
