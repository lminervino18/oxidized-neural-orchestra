use std::mem;

use ndarray::{linalg, prelude::*};

use crate::arch::activations::ActFn;

#[derive(Clone)]
pub struct Dense {
    dim: (usize, usize),
    size: usize,
    act_fn: Option<ActFn>,

    // Forward metadata
    x: Array2<f32>,
    z: Array2<f32>,
    a: Array2<f32>,

    // Backward metadata
    // TODO: hacer que el sequential tenga el delta y lo vaya pasando
    d: Array2<f32>,
}

impl Dense {
    pub fn new(dim: (usize, usize), act_fn: Option<ActFn>) -> Self {
        let size = (dim.0 + 1) * dim.1;
        let zeros = Array2::zeros((1, dim.1));

        Self {
            dim,
            size,
            act_fn,
            x: zeros.clone(),
            z: zeros.clone(),
            a: zeros.clone(),
            d: zeros,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

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

    fn view_params<'a>(&self, params: &'a [f32]) -> (ArrayView2<'a, f32>, ArrayView1<'a, f32>) {
        let w_size = self.size - self.dim.1;

        let weights = ArrayView2::from_shape(self.dim, &params[..w_size]).unwrap();
        let biases = ArrayView1::from_shape(self.dim.1, &params[w_size..]).unwrap();

        (weights, biases)
    }

    fn reshape(shape: (usize, usize), arr: Array2<f32>) -> Array2<f32> {
        let size = shape.0 * shape.1;

        let (mut v, Some(0)) = arr.into_raw_vec_and_offset() else {
            // TODO: ver de arreglar esto
            panic!("wtf, no es 0 el offset");
        };

        if let Some(additional) = size.checked_sub(v.len()) {
            v.reserve(additional);
            unsafe { v.set_len(size) };
        }

        Array2::from_shape_vec(shape, v).unwrap()
    }

    pub fn forward(&mut self, params: &[f32], x: ArrayView2<f32>) -> ArrayView2<'_, f32> {
        let (w, b) = self.view_params(params);

        let shape = (x.nrows(), self.dim.1);
        self.z = Self::reshape(shape, mem::take(&mut self.z));
        self.a = Self::reshape(shape, mem::take(&mut self.a));

        linalg::general_mat_mul(1.0, &x, &w, 0.0, &mut self.z);
        self.z += &b;

        self.x = x.to_owned();

        let Some(ref act_fn) = self.act_fn else {
            return self.z.view();
        };

        self.a.zip_mut_with(&self.z, |a, &z| *a = act_fn.f(z));
        self.a.view()
    }

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
        self.d = Self::reshape((d.nrows(), w.nrows()), mem::take(&mut self.d));
        linalg::general_mat_mul(1.0, &d, &w.t(), 0.0, &mut self.d);

        self.d.view_mut()
    }
}
