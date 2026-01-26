use ndarray::{linalg, Array1, Array2, ArrayView1, ArrayView2, Axis};

pub struct Dense {
    dim: (usize, usize),

    // Forward metadata
    x: Option<Array2<f32>>,

    // Backward metadata
    dw: Array2<f32>,
    db: Array1<f32>,
}

impl Dense {
    pub fn new(dim: (usize, usize)) -> Self {
        Self {
            dim,
            x: None,
            dw: Array2::zeros(dim),
            db: Array1::zeros(dim.1),
        }
    }

    fn view_params<'a>(&self, params: &'a [f32]) -> (ArrayView2<'a, f32>, ArrayView1<'a, f32>) {
        let (dim_in, dim_out) = self.dim;

        let w_range = ..params.len() - dim_out;
        let b_range = params.len() - dim_out..params.len();

        let weights = ArrayView2::from_shape((dim_in, dim_out), &params[w_range]).unwrap();
        let biases = ArrayView1::from_shape(dim_out, &params[b_range]).unwrap();

        (weights, biases)
    }

    pub fn forward(&mut self, params: &mut &[f32], x: Array2<f32>) -> Array2<f32> {
        let (dim_in, dim_out) = self.dim;

        let (weights, biases) = self.view_params(params);
        let offset = (dim_in + 1) * dim_out;
        *params = &params[offset..];

        let z = x.dot(&weights) + &biases;
        self.x = Some(x);
        z
    }

    pub fn backward(&mut self, params: &mut &[f32], d: Array2<f32>) -> Array2<f32> {
        let (dim_in, dim_out) = self.dim;

        let offset = params.len() - (dim_in + 1) * dim_out;
        let (weights, _) = self.view_params(&params[offset..]);
        *params = &params[offset..];

        // TODO: sacar el unwrap y devolver un layer::Result
        let x = self.x.take().unwrap();
        linalg::general_mat_mul(1.0, &x.t(), &d, 0.0, &mut self.dw);

        let inv_batch_size = 1.0 / d.nrows() as f32;
        self.db.view_mut().assign(&d.sum_axis(Axis(0)));
        self.db.mapv_inplace(|b| b * inv_batch_size);

        // TODO: ver de no alocar `d`s todo el tiempo
        weights.t().dot(&d)
    }
}
