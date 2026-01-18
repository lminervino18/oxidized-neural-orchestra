use super::WeightGen;

/// A weight generator that delegates the generation to a chain of weight generators.
///
/// This becomes handy whenever one wants a different weight generator for each layer of the model.
/// Using this wrapper one can configure a chain of generators such that each knows how much weights
/// to generate.
pub struct ChainedWeightGen {
    weight_gens: Vec<Box<dyn WeightGen>>,
    curr: usize,
}

impl ChainedWeightGen {
    /// Creates a new `ChainedWeightGen` weight generator.
    ///
    /// # Arguments
    /// * `weight_gens` - A vec of potentially different weight generators.
    pub fn new(weight_gens: Vec<Box<dyn WeightGen>>) -> Self {
        Self {
            weight_gens,
            curr: 0,
        }
    }
}

impl WeightGen for ChainedWeightGen {
    fn sample(&mut self, n: usize) -> Option<Vec<f32>> {
        if self.curr == self.weight_gens.len() {
            return None;
        }

        match self.weight_gens[self.curr].sample(n) {
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
    use super::{super::ConstWeightGen, *};

    #[test]
    fn empty() {
        let mut weight_gen = ChainedWeightGen::new(vec![]);
        assert!(weight_gen.sample(1).is_none());
    }

    #[test]
    fn exact() {
        let weight_gens: Vec<Box<dyn WeightGen>> = vec![
            Box::new(ConstWeightGen::new(0., 5)),
            Box::new(ConstWeightGen::new(1., 5)),
        ];

        let mut weight_gen = ChainedWeightGen::new(weight_gens);
        let sample = weight_gen.sample(10).unwrap();
        let expected: Vec<_> = (0..10).map(|i| (i > 4) as u32 as f32).collect();

        assert_eq!(sample, expected);
        assert!(weight_gen.sample(1).is_none());
    }

    #[test]
    fn partial() {
        let weight_gens: Vec<Box<dyn WeightGen>> = vec![
            Box::new(ConstWeightGen::new(0., 1)),
            Box::new(ConstWeightGen::new(1., 3)),
        ];

        let mut weight_gen = ChainedWeightGen::new(weight_gens);

        let sample = weight_gen.sample(2).unwrap();
        assert_eq!(sample, [0., 1.]);

        let sample = weight_gen.sample(2).unwrap();
        assert_eq!(sample, [1., 1.]);
        assert!(weight_gen.sample(1).is_none());
    }

    #[test]
    fn recusive() {
        let inner_weight_gens: Vec<Box<dyn WeightGen>> = vec![
            Box::new(ConstWeightGen::new(1., 1)),
            Box::new(ConstWeightGen::new(2., 1)),
        ];

        let weight_gens: Vec<Box<dyn WeightGen>> = vec![
            Box::new(ConstWeightGen::new(0., 1)),
            Box::new(ChainedWeightGen::new(inner_weight_gens)),
            Box::new(ConstWeightGen::new(3., 1)),
        ];

        let mut weight_gen = ChainedWeightGen::new(weight_gens);
        let sample = weight_gen.sample(4).unwrap();

        assert_eq!(sample, [0., 1., 2., 3.]);
        assert!(weight_gen.sample(1).is_none());
    }
}
