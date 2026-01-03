//! Dataset primitives (placeholder for now).

#[derive(Debug, Clone)]
pub struct Sample {
    pub x: f32,
    pub y: f32,
}

/// Simple in-memory dataset used for tests and early integration.
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    xs: Vec<f32>,
    ys: Vec<f32>,
}

impl InMemoryDataset {
    pub fn new(xs: Vec<f32>, ys: Vec<f32>) -> Self {
        assert_eq!(xs.len(), ys.len());
        Self { xs, ys }
    }

    pub fn len(&self) -> usize {
        self.xs.len()
    }

    pub fn get(&self, i: usize) -> Sample {
        Sample { x: self.xs[i], y: self.ys[i] }
    }
}
