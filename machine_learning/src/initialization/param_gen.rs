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
    /// The next `n` (at most) parameters in the generator or `None` if exhausted.
    fn sample(&mut self, n: usize) -> Option<Vec<f32>>;

    /// Samples the ramaining parameters in the generator.
    ///
    /// # Returns
    /// All the remaining parameters in the generator or `None` if exhausted.
    fn sample_remaining(&mut self) -> Option<Vec<f32>>;
}
