use std::cell::RefCell;

use rand::Rng;
use rand_distr::{Distribution, Normal, NormalError, Uniform, uniform::Error as UniformError};

use super::WeightGen;

/// A weight generator that follows a certain probabilistic distribution.
pub struct RandWeightGen<'a, R: Rng, D: Distribution<f32>> {
    rng: &'a RefCell<R>,
    distribution: D,
    remaining: usize,
}

impl<'a, R: Rng, D: Distribution<f32>> RandWeightGen<'a, R, D> {
    /// Creates a new `RandWeightGen` weight generator.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `distribution` - The distribution to sample the random numbers from.
    /// * `limit` - The maximum amount of numbers to generate.
    pub fn new(rng: &'a RefCell<R>, distribution: D, limit: usize) -> Self {
        Self {
            rng,
            distribution,
            remaining: limit,
        }
    }
}

impl<'a, R: Rng> RandWeightGen<'a, R, Uniform<f32>> {
    /// Creates a new `RandWeightGen` weight generator with a uniform distribution.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `low` - The incluive lower limit.
    /// * `high` - The exclusive upper limit.
    ///
    /// # Returns
    /// An error if the range is invalid (low > high).
    pub fn uniform(
        rng: &'a RefCell<R>,
        limit: usize,
        low: f32,
        high: f32,
    ) -> Result<Self, UniformError> {
        Ok(Self::new(rng, Uniform::new(low, high)?, limit))
    }

    /// Creates a new `RandWeightGen` weight generator with an inclusive uniform distribution.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `low` - The inclusive lower limit.
    /// * `high` - The inclusive upper limit.
    ///
    /// # Returns
    /// An error if the range is invalid (low > high).
    pub fn uniform_inclusive(
        rng: &'a RefCell<R>,
        limit: usize,
        low: f32,
        high: f32,
    ) -> Result<Self, UniformError> {
        Ok(Self::new(rng, Uniform::new_inclusive(low, high)?, limit))
    }

    /// Creates a new `RandWeightGen` weight generator using Xavier uniform initialization.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    /// * `fan_out` - The number of output units in the weight tensor.
    ///
    /// # Returns
    /// An error if the calculated range is invalid.
    pub fn xavier_uniform(
        rng: &'a RefCell<R>,
        limit: usize,
        fan_in: usize,
        fan_out: usize,
    ) -> Result<Self, UniformError> {
        let range = (6. / (fan_in + fan_out) as f32).sqrt();
        Self::uniform(rng, limit, -range, range)
    }

    /// Creates a new `RandWeightGen` weight generator using LeCun uniform initialization.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    ///
    /// # Returns
    /// An error if the caluculated range is invalid.
    pub fn lecun_uniform(
        rng: &'a RefCell<R>,
        limit: usize,
        fan_in: usize,
    ) -> Result<Self, UniformError> {
        let range = (3. / fan_in as f32).sqrt();
        Self::uniform(rng, limit, -range, range)
    }
}

impl<'a, R: Rng> RandWeightGen<'a, R, Normal<f32>> {
    /// Creates a new `RandWeightGen` weight generator with a normal distribution.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `mean` - The mean of the distribution.
    /// * `std_dev` - The standard deviation of the distribution.
    ///
    /// # Returns
    /// An error if `std_dev` is not finite (Nan or infinite).
    pub fn normal(
        rng: &'a RefCell<R>,
        limit: usize,
        mean: f32,
        std_dev: f32,
    ) -> Result<Self, NormalError> {
        Ok(Self::new(rng, Normal::new(mean, std_dev)?, limit))
    }

    /// Creates a new `RandWeightGen` weight generator using Kaiming normal initialization.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    ///
    /// # Returns
    /// An error if the calculated standard deviation is not finite (Nan or infinite).
    pub fn kaiming(rng: &'a RefCell<R>, limit: usize, fan_in: usize) -> Result<Self, NormalError> {
        let std_dev = (2. / fan_in as f32).sqrt();
        Self::normal(rng, limit, 0., std_dev)
    }

    /// Creates a new `RandWeightGen` weight generator using Xavier normal initialization.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    /// * `fan_out` - The number of output units in the weight tensor.
    ///
    /// # Returns
    /// An error if the calculated standard deviation is not finite (Nan or infinite).
    pub fn xavier(
        rng: &'a RefCell<R>,
        limit: usize,
        fan_in: usize,
        fan_out: usize,
    ) -> Result<Self, NormalError> {
        Self::kaiming(rng, limit, fan_in + fan_out)
    }

    /// Creates a new `RandWeightGen` weight generator using LeCun normal initialization.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    ///
    /// # Returns
    /// An error if the calculated standard deviation is not finite (Nan or infinite).
    pub fn lecun(rng: &'a RefCell<R>, limit: usize, fan_in: usize) -> Result<Self, NormalError> {
        let std_dev = (1. / fan_in as f32).sqrt();
        Self::normal(rng, limit, 0., std_dev)
    }
}

impl<'a, R: Rng, D: Distribution<f32>> WeightGen for RandWeightGen<'a, R, D> {
    fn sample(&mut self, mut n: usize) -> Option<Vec<f32>> {
        if self.remaining == 0 {
            return None;
        }

        n = n.min(self.remaining);
        self.remaining -= n;

        let mut rng = self.rng.borrow_mut();
        let sample = (0..n).map(|_| self.distribution.sample(&mut rng)).collect();
        Some(sample)
    }

    fn remaining(&self) -> usize {
        self.remaining
    }
}
