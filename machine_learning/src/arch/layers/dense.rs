use ndarray::{linalg, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

use crate::arch::activations::ActFn;

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
        let zeros = Array2::zeros((0, dim.1));

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

    pub fn forward(&mut self, params: &[f32], x: ArrayView2<f32>) -> ArrayView2<'_, f32> {
        let (w, b) = self.view_params(params);

        self.x = x.to_owned();
        self.z = x.dot(&w) + &b;

        if let Some(ref act_fn) = self.act_fn {
            self.a = self.z.mapv(|z| act_fn.f(z));
            return self.a.view();
        }

        self.z.view()
    }

    pub fn backward(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        d: ArrayView2<f32>,
    ) -> ArrayView2<'_, f32> {
        // TODO: hacer que el d que entra sea mutable
        let mut d = d.to_owned();

        if let Some(act_fn) = &self.act_fn {
            d *= &self.z.mapv(|z| act_fn.df(z));
        }

        let (mut dw, mut db) = self.view_grad(grad);
        let inv_batch_size = 1.0 / d.nrows() as f32;

        linalg::general_mat_mul(inv_batch_size, &self.x.t(), &d, 0.0, &mut dw);
        db.view_mut().assign(&d.sum_axis(Axis(0)));
        db.mapv_inplace(|b| b * inv_batch_size);

        let (w, _) = self.view_params(params);
        self.d = d.dot(&w.t());
        self.d.view()
    }
}
