/// A `ParamGen` generates values for the initial state of the model's parameters.
pub trait ParamGen {
    /// Should sample at most `n` parameters.
    ///
    /// # Arguments
    /// * `n` - The upper limit of samples to generate.
    ///
    /// # Returns
    /// An option whether the generator is exhausted.
    fn sample(&mut self, n: usize) -> Option<Vec<f32>>;
}
