use super::ParamGen;

/// A parameter generator that delegates the generation to a chain of parameter generators.
///
/// This becomes handy whenever one wants a different parameter generator for each layer of the model.
/// Using this wrapper one can configure a chain of generators such that each knows how much parameters to generate.
pub struct ChainedParamGen {
    param_gens: Vec<Box<dyn ParamGen>>,
    curr: usize,
}

impl ChainedParamGen {
    /// Creates a new `ChainedParamGen` parameter generator.
    ///
    /// # Arguments
    /// * `param_gens` - A vec of potentially different parameter generators.
    ///
    /// # Returns
    /// A new `ChainedParamGen` instance.
    pub fn new(param_gens: Vec<Box<dyn ParamGen>>) -> Self {
        Self {
            param_gens,
            curr: 0,
        }
    }
}

impl ParamGen for ChainedParamGen {
    fn sample(&mut self, n: usize) -> Option<Vec<f32>> {
        if self.curr == self.param_gens.len() {
            return None;
        }

        match self.param_gens[self.curr].sample(n) {
            Some(sample) if sample.len() == n => Some(sample),
            Some(mut sample) => {
                self.curr += 1;

                if let Some(next_sample) = self.sample(n - sample.len()) {
                    sample.extend(next_sample);
                }

                Some(sample)
            }
            None => {
                self.curr += 1;
                self.sample(n)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{super::ConstParamGen, *};

    #[test]
    fn empty() {
        let mut param_gen = ChainedParamGen::new(vec![]);
        assert!(param_gen.sample(1).is_none());
    }

    #[test]
    fn exact() {
        let param_gens: Vec<Box<dyn ParamGen>> = vec![
            Box::new(ConstParamGen::new(0., 5)),
            Box::new(ConstParamGen::new(1., 5)),
        ];

        let mut param_gen = ChainedParamGen::new(param_gens);
        let sample = param_gen.sample(10).unwrap();
        let expected: Vec<_> = (0..10).map(|i| (i > 4) as u32 as f32).collect();

        assert_eq!(sample, expected);
        assert!(param_gen.sample(1).is_none());
    }

    #[test]
    fn partial() {
        let param_gens: Vec<Box<dyn ParamGen>> = vec![
            Box::new(ConstParamGen::new(0., 1)),
            Box::new(ConstParamGen::new(1., 3)),
        ];

        let mut param_gen = ChainedParamGen::new(param_gens);

        let sample = param_gen.sample(2).unwrap();
        assert_eq!(sample, [0., 1.]);

        let sample = param_gen.sample(2).unwrap();
        assert_eq!(sample, [1., 1.]);
        assert!(param_gen.sample(1).is_none());
    }

    #[test]
    fn recusive() {
        let inner_param_gens: Vec<Box<dyn ParamGen>> = vec![
            Box::new(ConstParamGen::new(1., 1)),
            Box::new(ConstParamGen::new(2., 1)),
        ];

        let param_gens: Vec<Box<dyn ParamGen>> = vec![
            Box::new(ConstParamGen::new(0., 1)),
            Box::new(ChainedParamGen::new(inner_param_gens)),
            Box::new(ConstParamGen::new(3., 1)),
        ];

        let mut param_gen = ChainedParamGen::new(param_gens);
        let sample = param_gen.sample(4).unwrap();

        assert_eq!(sample, [0., 1., 2., 3.]);
        assert!(param_gen.sample(1).is_none());
    }
}
