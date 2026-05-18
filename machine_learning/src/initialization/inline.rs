use crate::initialization::ParamGen;

pub struct InlineParamGen {
    params: Vec<f32>,
    curr: usize,
}

impl InlineParamGen {
    /// Creates a new `InlineParamGen` parameter generator.
    ///
    /// # Args
    /// * `params` - The params to later yield.
    ///
    /// # Returns
    /// A new `InlineParamGen` instance.
    pub fn new(params: Vec<f32>) -> Self {
        Self { params, curr: 0 }
    }
}

impl ParamGen for InlineParamGen {
    fn size(&self) -> usize {
        self.params.len().saturating_sub(self.curr)
    }

    fn sample(&mut self, mut n: usize) -> Option<Vec<f32>> {
        n = n.min(self.size());

        if n == 0 {
            return None;
        }

        let params = &self.params[self.curr..self.curr + n];
        self.curr += n;
        Some(params.to_vec())
    }

    fn sample_remaining(&mut self) -> Option<Vec<f32>> {
        self.sample(self.size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let mut param_gen = InlineParamGen::new(vec![]);
        assert!(param_gen.sample(1).is_none());
    }

    #[test]
    fn exact() {
        const SIZE: usize = 10;

        let inline: Vec<_> = (0..SIZE).map(|i| i as f32).collect();
        let mut param_gen = InlineParamGen::new(inline.clone());
        let sample = param_gen.sample(SIZE).unwrap();

        assert_eq!(sample, inline);
        assert!(param_gen.sample(1).is_none());
    }

    #[test]
    fn partial() {
        const SIZE: usize = 10;

        let inline: Vec<_> = (0..SIZE).map(|i| i as f32).collect();
        let mut param_gen = InlineParamGen::new(inline.clone());

        let sample = param_gen.sample(7).unwrap();
        assert_eq!(sample, &inline[..7]);

        let sample = param_gen.sample(7).unwrap();
        assert_eq!(sample, &inline[7..]);

        assert!(param_gen.sample(1).is_none());
    }
}
