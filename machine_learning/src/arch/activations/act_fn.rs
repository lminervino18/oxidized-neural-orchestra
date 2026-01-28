use super::Sigmoid;

#[derive(Clone)]
pub enum ActFn {
    Sigmoid(Sigmoid),
    Step(Step),
}
use crate::arch::activations::Step;
use ActFn::*;

impl ActFn {
    pub fn sigmoid(amp: f32) -> Self {
        Sigmoid(Sigmoid::new(amp))
    }

    pub fn step(top: f32, bottom: f32, tresh: f32) -> Self {
        Step(Step::new(top, bottom, tresh))
    }

    pub fn f(&self, x: f32) -> f32 {
        match self {
            Sigmoid(a) => a.f(x),
            Step(a) => a.f(x),
        }
    }

    pub fn df(&self, x: f32) -> f32 {
        match self {
            Sigmoid(a) => a.df(x),
            Step(a) => a.df(x),
        }
    }
}
