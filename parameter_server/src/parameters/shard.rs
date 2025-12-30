use parking_lot::{Mutex, RwLock};

use super::optimization::Optimizer;

/// A buffer for accumulating gradients and weights across multiple threads.
#[derive(Debug)]
pub struct ParameterShard<O: Optimizer> {
    grads: [Mutex<Box<[f32]>>; 2],
    weights: RwLock<Box<[f32]>>,
    optimizer: Mutex<O>,
}

impl<O: Optimizer> ParameterShard<O> {
    /// Creates a new `ParameterShard`.
    ///
    /// # Arguments
    /// * `params` - The amount of parameters to hold.
    /// * `optimizer` - The optimization algorithm.
    pub fn new(params: usize, optimizer: O) -> Self {
        Self {
            grads: [
                Mutex::new(vec![0.; params].into_boxed_slice()),
                Mutex::new(vec![0.; params].into_boxed_slice()),
            ],
            weights: RwLock::new(vec![0.; params].into_boxed_slice()),
            optimizer: Mutex::new(optimizer),
        }
    }

    /// Accumulates `grad` into the active gradient.
    ///
    /// # Arguments
    /// * `active_idx` - The index of the active gradient, must be `0` or `1`.
    /// * `grad` - The gradient to accumulate to the active gradient.
    pub fn accumulate(&self, active_idx: usize, grad: &[f32]) {
        let mut active_grad = self.grads[active_idx].lock();
        active_grad
            .iter_mut()
            .zip(grad)
            .for_each(|(acc, g)| *acc += g);
    }

    /// Updates the weights using the frozen gradient via the optimizer and clears it.
    ///
    /// # Arguments
    /// * `frozen_idx` - The index of the frozen gradient, must be `0` or `1`.
    pub fn update_weights(&self, frozen_idx: usize) {
        let mut weights = self.weights.write();
        let mut grad = self.grads[frozen_idx].lock();
        self.optimizer.lock().update_weights(&grad, &mut weights);
        grad.fill(0.);
    }

    /// Copies the shard's inner weights into the provided destination buffer.
    ///
    /// # Arguments
    /// * `out` - A mutable slice where the weights will be copied.
    ///
    /// # Panics
    /// If the length of `out` doesn't match with the length of the shard.
    pub fn pull_weights(&self, out: &mut [f32]) {
        let weights = self.weights.read();
        out.copy_from_slice(&weights);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddOptimizer;

    impl Optimizer for AddOptimizer {
        fn update_weights(&mut self, grad: &[f32], weights: &mut [f32]) {
            weights.iter_mut().zip(grad).for_each(|(w, g)| *w += g);
        }
    }

    #[test]
    fn test_accumulation_and_update() {
        let shard = ParameterShard::new(3, AddOptimizer);

        shard.accumulate(0, &[1.0, 2.0, 3.0]);
        shard.accumulate(0, &[1.0, 1.0, 1.0]);

        {
            let grad0 = shard.grads[0].lock();
            assert_ne!(**grad0, [0., 0., 0.]);
        }
        {
            let grad1 = shard.grads[1].lock();
            assert_eq!(**grad1, [0., 0., 0.]);
        }

        shard.update_weights(0);

        let mut out = [0.; 3];
        shard.pull_weights(&mut out);
        assert_eq!(out, [2., 3., 4.]);
    }

    #[test]
    fn test_double_buffering_flow() {
        let shard = ParameterShard::new(1, AddOptimizer);

        shard.accumulate(0, &[10.]);
        shard.accumulate(1, &[5.]);
        shard.update_weights(0);

        let mut out = [0.];
        shard.pull_weights(&mut out);
        assert_eq!(out, [10.]);

        shard.update_weights(1);
        shard.pull_weights(&mut out);
        assert_eq!(out, [15.]);
    }
}
