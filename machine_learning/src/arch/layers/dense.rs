use ndarray::{linalg, prelude::*};

use super::InplaceReshape;
use crate::arch::activations::ActFn;

/// Optimizations:
///   1. Find a way to not copy `x` in each `Dense::forward` call.
///   2. Find a way to sum up the rows of `d` in `Dense::backward` in parallel to write them to `b`.
#[derive(Clone)]
pub struct Dense {
    dim: (usize, usize),
    act_fn: Option<ActFn>,
    size: usize,

    // Forward metadata
    x: Array2<f32>,
    z: Array2<f32>,
    a: Array2<f32>,

    // Backward metadata
    d: Array2<f32>,
}

impl Dense {
    // TODO: docstring
    pub fn new(dim: (usize, usize), act_fn: Option<ActFn>) -> Self {
        let zeros = Array2::zeros((1, 1));

        Self {
            dim,
            size: (dim.0 + 1) * dim.1,
            act_fn,
            x: zeros.clone(),
            z: zeros.clone(),
            a: zeros.clone(),
            d: zeros,
        }
    }

    /// Returns the size of this layer.
    ///
    /// # Returns
    /// The amount of parameters this layer has.
    pub fn size(&self) -> usize {
        self.size
    }

    // TODO: docstring
    pub fn forward(&mut self, params: &[f32], x: ArrayView2<f32>) -> ArrayView2<'_, f32> {
        let (w, b) = self.view_params(params);
        let shape = (x.nrows(), self.dim.1);

        self.z = self.z.into_reshape(shape);
        linalg::general_mat_mul(1.0, &x, &w, 0.0, &mut self.z);
        self.z += &b;

        self.x = x.to_owned();

        let Some(ref act_fn) = self.act_fn else {
            return self.z.view();
        };

        self.a = self.a.into_reshape(shape);
        self.a.zip_mut_with(&self.z, |a, &z| *a = act_fn.f(z));
        self.a.view()
    }

    // TODO: docstring
    pub fn backward(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        mut d: ArrayViewMut2<f32>,
    ) -> ArrayViewMut2<'_, f32> {
        if let Some(act_fn) = &self.act_fn {
            d.zip_mut_with(&self.z, |d, &z| *d *= act_fn.df(z));
        }

        let (mut dw, mut db) = self.view_grad(grad);
        linalg::general_mat_mul(1.0, &self.x.t(), &d, 0.0, &mut dw);
        db.view_mut().assign(&d.sum_axis(Axis(0)));

        let (w, _) = self.view_params(params);
        self.d = self.d.into_reshape((d.nrows(), w.nrows()));
        linalg::general_mat_mul(1.0, &d, &w.t(), 0.0, &mut self.d);

        self.d.view_mut()
    }

    /// Gives a view of the raw gradient slice as the delta weights and delta biases of this layer.
    ///
    /// # Arguments
    /// * `grad` - A gradient slice.
    ///
    /// # Returns
    /// A tuple containing the delta weights and delta biases.
    fn view_grad<'a>(
        &self,
        grad: &'a mut [f32],
    ) -> (ArrayViewMut2<'a, f32>, ArrayViewMut1<'a, f32>) {
        let w_size = self.size - self.dim.1;
        let (dw_raw, db_raw) = grad.split_at_mut(w_size);
        let dw = ArrayViewMut2::from_shape(self.dim, dw_raw).unwrap();
        let db = ArrayViewMut1::from_shape(self.dim.1, db_raw).unwrap();
        (dw, db)
    }

    /// Gives a view of the raw parameter slice as the weights and biases of this layer.
    ///
    /// # Arguments
    /// * `params` - A slice of parameters.
    ///
    /// # Returns
    /// A tuple containing the weights and biases.
    fn view_params<'a>(&self, params: &'a [f32]) -> (ArrayView2<'a, f32>, ArrayView1<'a, f32>) {
        let w_size = self.size - self.dim.1;
        let weights = ArrayView2::from_shape(self.dim, &params[..w_size]).unwrap();
        let biases = ArrayView1::from_shape(self.dim.1, &params[w_size..]).unwrap();
        (weights, biases)
    }
}
