pub trait WeightGen {
    /// Should sample at most `n` weights.
    ///
    /// # Arguments
    /// * `n` - The upper limit of samples to generate.
    ///
    /// # Returns
    /// An option whether the generator is exhausted.
    fn sample(&mut self, n: usize) -> Option<Vec<f32>>;

    /// Should return the amount of remaining weights this generator can still generate.
    ///
    /// # Returns
    /// The amount of remaining weights.
    fn remaining(&self) -> usize;
}
