use std::cell::UnsafeCell;

use crate::{
    optimization::Optimizer,
    storage::{Result, SizeMismatchErr},
};

/// A buffer for accumulating parameters across multiple threads without using locks.
///
/// It embraces race conditions, let's workers update the parameters simultaniously.
pub struct WildShard<O: Optimizer> {
    nparams: usize,
    params: UnsafeCell<Box<[f32]>>,
    optimizer: UnsafeCell<O>,
}

unsafe impl<O: Optimizer> Send for WildShard<O> {}
unsafe impl<O: Optimizer> Sync for WildShard<O> {}

impl<O: Optimizer> WildShard<O> {
    /// Creates a new `WildShard` parameter shard.
    ///
    /// # Arguments
    /// * `params` - The initial state of the parameters.
    /// * `optimizer` - The optimization algorithm.
    ///
    /// # Returns
    /// A new `BlockingShard` instance.
    pub fn new(params: Vec<f32>, optimizer: O) -> Self {
        Self {
            nparams: params.len(),
            params: UnsafeCell::new(params.into_boxed_slice()),
            optimizer: UnsafeCell::new(optimizer),
        }
    }

    /// Updates the parameters using a new gradient via the optimizer.
    ///
    /// # Arguments
    /// * `grad` - The gradient to accumulate to the shard's parameters.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if `grad` isn't the same size as this shard.
    pub fn update_params(&self, grad: &[f32]) -> Result<()> {
        if self.nparams != grad.len() {
            return Err(SizeMismatchErr);
        }

        // SAFETY: Both params and optimizer are pinned to memory during the `Shard`'s life. It will
        //         be maintaind valid and initialized during this method's execution.
        //
        //        For this particular shard implementation we're embracing race conditions.
        let params = unsafe { &mut *self.params.get() };
        let optimizer = unsafe { &mut *self.optimizer.get() };

        // SAFETY: Both grad and params have the same length.
        optimizer.update_params(grad, params).unwrap();
        Ok(())
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

        // SAFETY: Params is pinned to memory during the `Shard`'s life. It will
        //         be maintaind valid and initialized during this method's execution.
        let params = unsafe { &*self.params.get() };

        // SAFETY: We checked both lengths match just above.
        out.copy_from_slice(params);
        Ok(())
    }
}
