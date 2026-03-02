use super::Sigmoid;

/// A type of layer activation function.
#[derive(Clone)]
pub enum ActFn {
    Sigmoid(Sigmoid),
}
use ActFn::*;

impl ActFn {
    /// Creates a new `Sigmoid` activation function.
    ///
    /// # Arguments
    /// * `amp` - The amplitude of the function.
    ///
    /// # Returns
    /// A new `Sigmoid` instance.
    pub fn sigmoid(amp: f32) -> Self {
        Sigmoid(Sigmoid::new(amp))
    }

    /// The proper function call.
    ///
    /// # Arguments
    /// * `x` - The input to the function.
    ///
    /// # Returns
    /// The evaluation of the input.
    pub fn f(&self, x: f32) -> f32 {
        match self {
            Sigmoid(a) => a.f(x),
        }
    }

    /// The derivative of the function.
    ///
    /// # Arguments
    /// * `x` - The input to the function.
    ///
    /// # Returns
    /// The evaluation of th input.
    pub fn df(&self, x: f32) -> f32 {
        match self {
            Sigmoid(a) => a.df(x),
        }
    }
}
