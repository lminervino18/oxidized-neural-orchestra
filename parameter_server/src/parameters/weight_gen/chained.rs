use super::WeightGen;

/// A weight generator that delegates the generation to a chain of weight generators.
///
/// This becomes handy whenever one wants a different weight generator for each layer of the model.
/// Using this wrapper one can configure a chain of generators such that each knows how much weights
/// to generate.
pub struct ChainedWeightGen {
    weight_gens: Vec<Box<dyn WeightGen>>,
    curr: usize,
    remaining: usize,
}

impl ChainedWeightGen {
    /// Creates a new `ChainedWeightGen` weight generator.
    ///
    /// # Arguments
    /// * `weight_gens` - A vec of potentially different weight generators.
    pub fn new(weight_gens: Vec<Box<dyn WeightGen>>) -> Self {
        let remaining = weight_gens
            .iter()
            .map(|weight_gen| weight_gen.remaining())
            .sum();

        Self {
            weight_gens,
            remaining,
            curr: 0,
        }
    }
}

impl WeightGen for ChainedWeightGen {
    fn sample(&mut self, n: usize) -> Option<Vec<f32>> {
        if self.curr == self.weight_gens.len() || self.remaining == 0 {
            return None;
        }

        match self.weight_gens[self.curr].sample(n) {
            Some(sample) if sample.len() == n => {
                self.remaining -= sample.len();
                Some(sample)
            }
            Some(mut sample) => {
                self.remaining -= sample.len();
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

    fn remaining(&self) -> usize {
        self.remaining
    }
}
