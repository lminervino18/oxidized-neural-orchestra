/// A `ParamGen` generates values for the initial state of the model's parameters.
pub trait ParamGen {
    /// Returns the amount of remaining numbers to be generated.
    ///
    /// # Returns
    /// The amount of numbers left to generate.
    fn size(&self) -> usize;

    /// Should sample at most `n` parameters.
    ///
    /// # Args
    /// * `n` - The upper limit of samples to generate.
    ///
    /// # Returns
    /// An option whether the generator is exhausted.
    fn sample(&mut self, n: usize) -> Option<Vec<f32>>;
}
