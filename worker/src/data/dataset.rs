/// A single supervised sample (x, y).
///
/// Baseline model (LinearRegression1D) uses scalar x and y.
/// Later we can generalize to multi-dimensional features/tensors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub x: f32,
    pub y: f32,
}

/// A minimal in-memory dataset.
///
/// Design goals:
/// - deterministic and test-friendly
/// - small API surface
/// - compatibility with current DataLoader (`get`)
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    xs: Vec<f32>,
    ys: Vec<f32>,
}

impl InMemoryDataset {
    /// Creates a new dataset from owned buffers.
    ///
    /// # Panics
    /// - if `xs.len() != ys.len()`
    /// - if `xs` is empty
    pub fn new(xs: Vec<f32>, ys: Vec<f32>) -> Self {
        assert_eq!(xs.len(), ys.len(), "xs and ys must have same length");
        assert!(!xs.is_empty(), "dataset must be non-empty");
        Self { xs, ys }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// Returns the sample at `idx` (panics if out of bounds).
    #[inline]
    pub fn sample(&self, idx: usize) -> Sample {
        Sample {
            x: self.xs[idx],
            y: self.ys[idx],
        }
    }

    /// Returns the sample at `idx` (panics if out of bounds).
    ///
    /// Kept as `get` for compatibility with the current DataLoader implementation.
    #[inline]
    pub fn get(&self, idx: usize) -> Sample {
        self.sample(idx)
    }

    #[inline]
    pub fn xs(&self) -> &[f32] {
        &self.xs
    }

    #[inline]
    pub fn ys(&self) -> &[f32] {
        &self.ys
    }
}

/// A minimal owned batch of training data.
///
/// Today it's owned (`Vec`) for simplicity.
/// In a later iteration, DataLoader can yield borrowed batches for zero-copy.
#[derive(Debug, Clone)]
pub struct Batch {
    pub xs: Vec<f32>,
    pub ys: Vec<f32>,
}

impl Batch {
    /// # Panics
    /// - if `xs.len() != ys.len()`
    /// - if `xs` is empty
    pub fn new(xs: Vec<f32>, ys: Vec<f32>) -> Self {
        assert_eq!(xs.len(), ys.len(), "xs and ys must have same length");
        assert!(!xs.is_empty(), "batch must be non-empty");
        Self { xs, ys }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.xs.len()
    }
}

/// Borrowed batch view (zero-copy).
#[derive(Debug, Clone, Copy)]
pub struct BatchRef<'a> {
    pub xs: &'a [f32],
    pub ys: &'a [f32],
}

impl<'a> BatchRef<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.xs.len()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_basic() {
        let ds = InMemoryDataset::new(vec![1.0, 2.0], vec![3.0, 5.0]);
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.get(0), Sample { x: 1.0, y: 3.0 });
        assert_eq!(ds.sample(1), Sample { x: 2.0, y: 5.0 });
    }

    #[test]
    fn batch_basic() {
        let b = Batch::new(vec![1.0, 2.0], vec![3.0, 5.0]);
        assert_eq!(b.len(), 2);
    }
}
