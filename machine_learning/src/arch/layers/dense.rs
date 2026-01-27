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

        // TODO: deberíamos liberar la memoria si los xs van a ser más chicos
        if let Some(_) = x.nrows().checked_sub(self.z.nrows()) {
            self.x = Array2::zeros((x.nrows(), self.dim.1));
            self.a = Array2::zeros((x.nrows(), self.dim.1));
            self.z = Array2::zeros((x.nrows(), self.dim.1));
        }

        linalg::general_mat_mul(1.0, &x, &w, 0.0, &mut self.z);
        self.z += &b;
        self.x = x.to_owned(); // TODO: ver de no allocar

        if let Some(act_fn) = &self.act_fn {
            self.a.zip_mut_with(&self.z, |a, &z| {
                *a = act_fn.df(z);
            });

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
        let (mut dw, mut db) = self.view_grad(grad);

        linalg::general_mat_mul(1.0, &self.x.t(), &d, 0.0, &mut dw);

        let inv_batch_size = 1.0 / d.nrows() as f32;
        db.view_mut().assign(&d.sum_axis(Axis(0)));
        db.mapv_inplace(|b| b * inv_batch_size);

        if let Some(additional) = d.nrows().checked_sub(self.d.nrows()) {
            self.d.reserve_rows(additional).unwrap();
        }

        let (w, _) = self.view_params(params);
        linalg::general_mat_mul(1.0, &d, &w.t(), 0.0, &mut self.d);

        if let Some(act_fn) = &self.act_fn {
            self.d.zip_mut_with(&self.z, |d, &z| *d *= act_fn.df(z));
        }

        self.d.view()
    }
}
