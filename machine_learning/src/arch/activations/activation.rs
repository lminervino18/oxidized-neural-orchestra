use super::Sigmoid;

pub enum ActFn {
    Sigmoid(Sigmoid),
}
use ActFn::*;

impl ActFn {
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
