use std::{cell::RefCell, rc::Rc};

use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use super::{Result, WeightGen};

/// A weight generator that follows a certain probabilistic distribution.
pub struct RandWeightGen<R: Rng, D: Distribution<f32>> {
    rng: Rc<RefCell<R>>,
    distribution: D,
    remaining: usize,
}

impl<R: Rng, D: Distribution<f32>> RandWeightGen<R, D> {
    /// Creates a new `RandWeightGen` weight generator.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `distribution` - The distribution to sample the random numbers from.
    /// * `limit` - The maximum amount of numbers to generate.
    pub fn new(rng: Rc<RefCell<R>>, distribution: D, limit: usize) -> Self {
        Self {
            rng,
            distribution,
            remaining: limit,
        }
    }
}

impl<R: Rng> RandWeightGen<R, Uniform<f32>> {
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
    pub fn uniform(rng: Rc<RefCell<R>>, limit: usize, low: f32, high: f32) -> Result<Self> {
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
        rng: Rc<RefCell<R>>,
        limit: usize,
        low: f32,
        high: f32,
    ) -> Result<Self> {
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
        rng: Rc<RefCell<R>>,
        limit: usize,
        fan_in: usize,
        fan_out: usize,
    ) -> Result<Self> {
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
    pub fn lecun_uniform(rng: Rc<RefCell<R>>, limit: usize, fan_in: usize) -> Result<Self> {
        let range = (3. / fan_in as f32).sqrt();
        Self::uniform(rng, limit, -range, range)
    }
}

impl<R: Rng> RandWeightGen<R, Normal<f32>> {
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
    pub fn normal(rng: Rc<RefCell<R>>, limit: usize, mean: f32, std_dev: f32) -> Result<Self> {
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
    pub fn kaiming(rng: Rc<RefCell<R>>, limit: usize, fan_in: usize) -> Result<Self> {
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
        rng: Rc<RefCell<R>>,
        limit: usize,
        fan_in: usize,
        fan_out: usize,
    ) -> Result<Self> {
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
    pub fn lecun(rng: Rc<RefCell<R>>, limit: usize, fan_in: usize) -> Result<Self> {
        let std_dev = (1. / fan_in as f32).sqrt();
        Self::normal(rng, limit, 0., std_dev)
    }
}

impl<R: Rng, D: Distribution<f32>> WeightGen for RandWeightGen<R, D> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn seeded_rng() -> Rc<RefCell<StdRng>> {
        Rc::new(RefCell::new(StdRng::seed_from_u64(42)))
    }

    #[test]
    fn empty() {
        const SIZE: usize = 0;
        let rng = seeded_rng();

        let mut weight_gen = RandWeightGen::normal(rng, SIZE, 0., 1.).unwrap();
        assert!(weight_gen.sample(1).is_none());
    }

    #[test]
    fn exact() {
        const SIZE: usize = 10;
        let rng = seeded_rng();

        let mut weight_gen = RandWeightGen::uniform(rng, SIZE, -1., 1.).unwrap();
        let sample = weight_gen.sample(SIZE).unwrap();

        assert_eq!(sample.len(), SIZE);
        assert!(weight_gen.sample(1).is_none());
    }

    #[test]
    fn partial() {
        let rng = seeded_rng();

        let mut weight_gen = RandWeightGen::normal(rng, 10, 0., 1.).unwrap();

        let sample = weight_gen.sample(7).unwrap();
        assert_eq!(sample.len(), 7);

        let sample = weight_gen.sample(7).unwrap();
        assert_eq!(sample.len(), 3);

        assert!(weight_gen.sample(1).is_none());
    }
}
