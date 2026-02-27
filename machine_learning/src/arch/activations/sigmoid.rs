use std::f32;

#[derive(Clone, Debug, Default)]
pub struct Sigmoid {
    amp: f32,
}

impl Sigmoid {
    pub fn new(amp: f32) -> Self {
        Self { amp }
    }

    pub fn f(&self, z: f32) -> f32 {
        self.amp / (1. + (-z).exp())
    }

    pub fn df(&self, z: f32) -> f32 {
        let amp = self.amp;

        (amp * (-z).exp()) / ((-z).exp() + 1.).powi(2)
    }
}
