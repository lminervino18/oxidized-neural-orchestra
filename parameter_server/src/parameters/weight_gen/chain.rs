use rand::Rng;

use super::WeightGen;

///
pub struct ChainWeightGen<R: Rng> {
    weight_gens: Vec<Box<dyn WeightGen<R>>>,
    curr: usize,
}

impl<R: Rng> ChainWeightGen<R> {
    /// Creates a new `ChainWeightGen` weight generator.
    ///
    /// # Arguments
    /// * `weight_gens` - A vec of potentially different weight generators.
    pub fn new(weight_gens: Vec<Box<dyn WeightGen<R>>>) -> Self {
        Self {
            weight_gens,
            curr: 0,
        }
    }
}

impl<R: Rng> WeightGen<R> for ChainWeightGen<R> {
    fn sample<'a>(&mut self, rng: &'a mut R, n: usize) -> Option<Vec<f32>> {
        if self.curr == self.weight_gens.len() {
            return None;
        }

        match self.weight_gens[self.curr].sample(rng, n) {
            Some(sample) if sample.len() == n => Some(sample),
            Some(mut sample) => {
                self.curr += 1;

                if let Some(next_sample) = self.sample(rng, n - sample.len()) {
                    sample.extend(next_sample);
                }

                Some(sample)
            }
            None => {
                self.curr += 1;
                self.sample(rng, n)
            }
        }
    }

    fn remaining(&self) -> usize {
        if self.curr == self.weight_gens.len() {
            return 0;
        }

        self.weight_gens[self.curr..]
            .iter()
            .map(|weight_gen| weight_gen.remaining())
            .sum()
    }
}
