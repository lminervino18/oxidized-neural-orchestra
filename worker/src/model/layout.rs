use std::ops::Range;

use super::spec::ModelSpec;

/// Maps a flat parameter buffer into named tensors/slices.
/// This is the core "offsets + shapes" mechanism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterLayout {
    pub w: Range<usize>,
    pub b: Range<usize>,
}

impl ParameterLayout {
    pub fn new(spec: ModelSpec) -> Self {
        match spec {
            ModelSpec::LinearRegression1D => {
                // Flat layout: [w, b]
                Self {
                    w: 0..1,
                    b: 1..2,
                }
            }
        }
    }

    /// Sanity check: ranges must be in-bounds and non-overlapping for a given buffer size.
    pub fn validate(&self, total_params: usize) {
        assert!(self.w.start < self.w.end, "w range must be non-empty");
        assert!(self.b.start < self.b.end, "b range must be non-empty");
        assert!(self.w.end <= total_params, "w out of bounds");
        assert!(self.b.end <= total_params, "b out of bounds");

        // For this baseline layout, w then b. For future models, generalize overlap checks.
        assert!(self.w.end <= self.b.start, "layout ranges overlap");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::spec::ModelSpec;

    #[test]
    fn linreg_layout_is_valid() {
        let spec = ModelSpec::LinearRegression1D;
        let layout = ParameterLayout::new(spec);
        layout.validate(spec.num_params());
        assert_eq!(layout.w, 0..1);
        assert_eq!(layout.b, 1..2);
    }
}
