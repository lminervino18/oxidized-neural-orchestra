use rand::Rng;

pub trait WeightGen<R: Rng> {
    /// Should sample at most `n` weights.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `n` - The upper limit of samples to generate.
    ///
    /// # Returns
    /// An option whether the generator is exhausted.
    fn sample<'a>(&mut self, rng: &'a mut R, n: usize) -> Option<Vec<f32>>;

    /// Should return the amount of remaining weights this generator can still generate.
    ///
    /// # Returns
    /// The amount of remaining weights.
    fn remaining(&self) -> usize;
}
