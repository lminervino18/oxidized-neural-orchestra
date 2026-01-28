use ndarray::ArrayView2;

use super::Dense;
use crate::arch::activations::ActFn;

#[derive(Clone)]
pub enum Layer {
    Dense(Dense),
}
use Layer::*;

impl Layer {
    pub fn size(&self) -> usize {
        match self {
            Dense(l) => l.size(),
        }
    }

    pub fn dense(dim: (usize, usize), act_fn: ActFn) -> Self
where {
        Self::Dense(Dense::new(dim, act_fn.into()))
    }

    pub fn forward<'a>(&'a mut self, params: &[f32], x: ArrayView2<f32>) -> ArrayView2<'a, f32> {
        match self {
            Dense(l) => l.forward(params, x),
        }
    }

    pub fn backward<'a>(
        &'a mut self,
        params: &[f32],
        grad: &mut [f32],
        d: ArrayView2<f32>,
    ) -> ArrayView2<'a, f32> {
        match self {
            Dense(l) => l.backward(params, grad, d),
        }
    }
}
