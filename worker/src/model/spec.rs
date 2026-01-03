/// High-level model specification.
///
/// For now we only implement a minimal baseline model:
/// - Linear regression with 1 feature: y = w*x + b
///
/// The spec exists to keep the architecture extensible (MLP, CNN, etc.)
/// without changing worker loop or networking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSpec {
    /// y = w*x + b
    LinearRegression1D,
}

impl ModelSpec {
    /// Total number of parameters in the flat buffer for this model.
    pub fn num_params(self) -> usize {
        match self {
            ModelSpec::LinearRegression1D => 2, // w, b
        }
    }
}
