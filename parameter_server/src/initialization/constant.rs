use super::ParamGen;

/// A parameter generator that always generates the same value.
pub struct ConstParamGen {
    value: f32,
    remaining: usize,
}

impl ConstParamGen {
    /// Creates a new `ConstParamGen` parameter generator.
    ///
    /// # Arguments
    /// * `value` - The value to always generate.
    /// * `limit` - The maximum amount of times to generate that value.
    ///
    /// # Returns
    /// A new `ConstParamGen` instance.
    pub fn new(value: f32, limit: usize) -> Self {
        Self {
            value,
            remaining: limit,
        }
    }
}

impl ParamGen for ConstParamGen {
    fn sample(&mut self, mut n: usize) -> Option<Vec<f32>> {
        if self.remaining == 0 {
            return None;
        }

        n = n.min(self.remaining);
        self.remaining -= n;
        Some(vec![self.value; n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        const SIZE: usize = 0;

        let mut param_gen = ConstParamGen::new(1., SIZE);
        assert!(param_gen.sample(1).is_none());
    }

    #[test]
    fn exact() {
        const SIZE: usize = 10;

        let mut param_gen = ConstParamGen::new(1., SIZE);
        let sample = param_gen.sample(SIZE).unwrap();

        assert_eq!(sample, vec![1.; SIZE]);
        assert!(param_gen.sample(1).is_none());
    }

    #[test]
    fn partial() {
        let mut param_gen = ConstParamGen::new(1., 10);

        let sample = param_gen.sample(7).unwrap();
        assert_eq!(sample, vec![1.; 7]);

        let sample = param_gen.sample(7).unwrap();
        assert_eq!(sample, vec![1.; 3]);
        assert!(param_gen.sample(1).is_none());
    }
}
