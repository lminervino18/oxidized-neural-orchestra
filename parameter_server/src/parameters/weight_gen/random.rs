use rand::Rng;
use rand_distr::{Distribution, Normal, NormalError, Uniform, uniform::Error as UniformError};

use super::WeightGen;

/// A weight generator that follows a certain probabilistic distribution.
pub struct RandWeightGen<D: Distribution<f32>> {
    distribution: D,
    remaining: usize,
}

impl<D: Distribution<f32>> RandWeightGen<D> {
    /// Creates a new `RandWeightGen` weight generator.
    ///
    /// # Arguments
    /// * `distribution` - The distribution to sample the random numbers from.
    /// * `limit` - The maximum amount of numbers to generate.
    pub fn new(distribution: D, limit: usize) -> Self {
        Self {
            distribution,
            remaining: limit,
        }
    }
}

impl RandWeightGen<Uniform<f32>> {
    /// Creates a new `RandWeightGen` weight generator that always yields the same value.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `value` - The value to always generate.
    pub fn constant(limit: usize, value: f32) -> Self {
        // SAFETY: This range is always valid.
        Self::uniform_inclusive(limit, value, value).unwrap()
    }

    /// Creates a new `RandWeightGen` weight generator with a uniform distribution.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `low` - The incluive lower limit.
    /// * `high` - The exclusive upper limit.
    ///
    /// # Returns
    /// An error if the range is invalid (low > high).
    pub fn uniform(limit: usize, low: f32, high: f32) -> Result<Self, UniformError> {
        Ok(Self::new(Uniform::new(low, high)?, limit))
    }

    /// Creates a new `RandWeightGen` weight generator with an inclusive uniform distribution.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `low` - The inclusive lower limit.
    /// * `high` - The inclusive upper limit.
    ///
    /// # Returns
    /// An error if the range is invalid (low > high).
    pub fn uniform_inclusive(limit: usize, low: f32, high: f32) -> Result<Self, UniformError> {
        Ok(Self::new(Uniform::new_inclusive(low, high)?, limit))
    }

    /// Creates a new `RandWeightGen` weight generator using Xavier uniform initialization.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    /// * `fan_out` - The number of output units in the weight tensor.
    ///
    /// # Returns
    /// An error if the calculated range is invalid.
    pub fn xavier_uniform(
        limit: usize,
        fan_in: usize,
        fan_out: usize,
    ) -> Result<Self, UniformError> {
        let range = (6. / (fan_in + fan_out) as f32).sqrt();
        Self::uniform(limit, -range, range)
    }

    /// Creates a new `RandWeightGen` weight generator using LeCun uniform initialization.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    ///
    /// # Returns
    /// An error if the caluculated range is invalid.
    pub fn lecun_uniform(limit: usize, fan_in: usize) -> Result<Self, UniformError> {
        let range = (3. / fan_in as f32).sqrt();
        Self::uniform(limit, -range, range)
    }
}

impl RandWeightGen<Normal<f32>> {
    /// Creates a new `RandWeightGen` weight generator with a normal distribution.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `mean` - The mean of the distribution.
    /// * `std_dev` - The standard deviation of the distribution.
    ///
    /// # Returns
    /// An error if `std_dev` is not finite (Nan or infinite).
    pub fn normal(limit: usize, mean: f32, std_dev: f32) -> Result<Self, NormalError> {
        Ok(Self::new(Normal::new(mean, std_dev)?, limit))
    }

    /// Creates a new `RandWeightGen` weight generator using Kaiming normal initialization.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    ///
    /// # Returns
    /// An error if the calculated standard deviation is not finite (Nan or infinite).
    pub fn kaiming(limit: usize, fan_in: usize) -> Result<Self, NormalError> {
        let std_dev = (2. / fan_in as f32).sqrt();
        Self::normal(limit, 0., std_dev)
    }

    /// Creates a new `RandWeightGen` weight generator using Xavier normal initialization.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    /// * `fan_out` - The number of output units in the weight tensor.
    ///
    /// # Returns
    /// An error if the calculated standard deviation is not finite (Nan or infinite).
    pub fn xavier(limit: usize, fan_in: usize, fan_out: usize) -> Result<Self, NormalError> {
        Self::kaiming(limit, fan_in + fan_out)
    }

    /// Creates a new `RandWeightGen` weight generator using LeCun normal initialization.
    ///
    /// # Arguments
    /// * `limit` - The maximum amount of numbers to generate.
    /// * `fan_in` - The number of input units in the weight tensor.
    ///
    /// # Returns
    /// An error if the calculated standard deviation is not finite (Nan or infinite).
    pub fn lecun(limit: usize, fan_in: usize) -> Result<Self, NormalError> {
        let std_dev = (1. / fan_in as f32).sqrt();
        Self::normal(limit, 0., std_dev)
    }
}

impl<R: Rng, D: Distribution<f32>> WeightGen<R> for RandWeightGen<D> {
    fn sample<'a>(&mut self, rng: &'a mut R, mut n: usize) -> Option<Vec<f32>> {
        if self.remaining == 0 {
            return None;
        }

        n = n.min(self.remaining);
        self.remaining -= n;
        Some((0..n).map(|_| self.distribution.sample(rng)).collect())
    }

    fn remaining(&self) -> usize {
        self.remaining
    }
}
