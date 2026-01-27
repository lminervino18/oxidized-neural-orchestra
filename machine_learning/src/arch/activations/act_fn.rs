use super::Sigmoid;

pub enum ActFn {
    Sigmoid(Sigmoid),
}
use ActFn::*;

impl ActFn {
    pub fn sigmoid(amp: f32) -> Self {
        Sigmoid(Sigmoid::new(amp))
    }

    pub fn f(&self, x: f32) -> f32 {
        match self {
            Sigmoid(a) => a.f(x),
        }
    }

    pub fn df(&self, x: f32) -> f32 {
        match self {
            Sigmoid(a) => a.df(x),
        }
    }
}
