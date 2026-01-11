use super::WeightGen;

/// A weight generator that always generates the same value.
pub struct ConstWeightGen {
    value: f32,
    remaining: usize,
}

impl ConstWeightGen {
    /// Creates a new `ConstWeightGen` weight generator which always generates the same value.
    ///
    /// # Arguments
    /// * `value` - The value to always generate.
    /// * `limit` - The maximum amount of times to generate that value.
    pub fn new(value: f32, limit: usize) -> Self {
        Self {
            value,
            remaining: limit,
        }
    }
}

impl WeightGen for ConstWeightGen {
    fn sample(&mut self, mut n: usize) -> Option<Vec<f32>> {
        if self.remaining == 0 {
            return None;
        }

        n = n.min(self.remaining);
        self.remaining -= n;
        Some(vec![self.value; n])
    }
}
