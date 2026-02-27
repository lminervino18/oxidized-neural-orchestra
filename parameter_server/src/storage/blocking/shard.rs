use parking_lot::{Mutex, RwLock};

use crate::{
    optimization::Optimizer,
    storage::{Result, SizeMismatchErr},
};

/// A buffer for accumulating gradients and parameters across multiple threads using locks.
///
/// It implements a double-buffer strategy to let workers accumulate gradients in the active
/// buffer while the frozen buffer stays inactive to be able to reset it via `update_params`
/// without stoping other workers trying to accumulate more gradients.
#[derive(Debug)]
pub struct BlockingShard<O: Optimizer> {
    nparams: usize,
    grads: [Mutex<Box<[f32]>>; 2],
    params: RwLock<Box<[f32]>>,
    optimizer: Mutex<O>,
}

impl<O: Optimizer> BlockingShard<O> {
    /// Creates a new `BlockingShard` parameter shard.
    ///
    /// # Arguments
    /// * `params` - The initial state of the parameters.
    /// * `optimizer` - The optimization algorithm.
    ///
    /// # Returns
    /// A new `BlockingShard` instance.
    pub fn new(params: Vec<f32>, optimizer: O) -> Self {
        let nparams = params.len();

        Self {
            nparams,
            grads: [
                Mutex::new(vec![0.; nparams].into_boxed_slice()),
                Mutex::new(vec![0.; nparams].into_boxed_slice()),
            ],
            params: RwLock::new(params.into_boxed_slice()),
            optimizer: Mutex::new(optimizer),
        }
    }

    /// Accumulates `grad` into the active gradient.
    ///
    /// # Arguments
    /// * `active_idx` - The index of the active gradient, must be `0` or `1`.
    /// * `grad` - The gradient to accumulate to the active gradient.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if `grad` isn't the same size as this shard.
    pub fn accumulate(&self, active_idx: usize, grad: &[f32]) -> Result<()> {
        if self.nparams != grad.len() {
            return Err(SizeMismatchErr);
        }

        self.grads[active_idx]
            .lock()
            .iter_mut()
            .zip(grad)
            .for_each(|(acc, g)| *acc += g);

        Ok(())
    }

    /// Updates the parameters using the frozen gradient via the optimizer and clears it.
    ///
    /// # Arguments
    /// * `frozen_idx` - The index of the frozen gradient, must be `0` or `1`.
    pub fn update_params(&self, frozen_idx: usize) {
        let mut params = self.params.write();
        let mut grad = self.grads[frozen_idx].lock();

        // SAFETY: Both grad and params have the same length.
        self.optimizer
            .lock()
            .update_params(&grad, &mut params)
            .unwrap();

        grad.fill(0.);
    }

    /// Copies the shard's inner parameters into the provided destination buffer.
    ///
    /// # Arguments
    /// * `out` - A mutable slice where the parameters will be copied.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if `out` isn't the same size as this shard.
    pub fn pull_params(&self, out: &mut [f32]) -> Result<()> {
        if self.nparams != out.len() {
            return Err(SizeMismatchErr);
        }

        let params = self.params.read();
        out.copy_from_slice(&params);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddOptimizer;

    impl Optimizer for AddOptimizer {
        fn update_params(&mut self, grad: &[f32], params: &mut [f32]) -> Result<()> {
            params.iter_mut().zip(grad).for_each(|(w, g)| *w += g);
            Ok(())
        }
    }

    #[test]
    fn test_accumulation_and_update() {
        let shard = BlockingShard::new(vec![0.; 3], AddOptimizer);

        shard.accumulate(0, &[1.0, 2.0, 3.0]).unwrap();
        shard.accumulate(0, &[1.0, 1.0, 1.0]).unwrap();

        {
            let grad0 = shard.grads[0].lock();
            assert_ne!(**grad0, [0., 0., 0.]);
        }
        {
            let grad1 = shard.grads[1].lock();
            assert_eq!(**grad1, [0., 0., 0.]);
        }

        shard.update_params(0);

        let mut out = [0.; 3];
        shard.pull_params(&mut out).unwrap();
        assert_eq!(out, [2., 3., 4.]);
    }

    #[test]
    fn test_double_buffering_flow() {
        let shard = BlockingShard::new(vec![0.], AddOptimizer);

        shard.accumulate(0, &[10.]).unwrap();
        shard.accumulate(1, &[5.]).unwrap();
        shard.update_params(0);

        let mut out = [0.];
        shard.pull_params(&mut out).unwrap();
        assert_eq!(out, [10.]);

        shard.update_params(1);
        shard.pull_params(&mut out).unwrap();
        assert_eq!(out, [15.]);
    }
}
