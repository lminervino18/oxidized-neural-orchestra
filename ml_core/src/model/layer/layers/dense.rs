use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut2};

pub struct Dense {
    dim_in: usize,
    dim_out: usize,
}

impl Dense {
    pub fn new(dim: (usize, usize)) -> Self {
        Self {
            dim_in: dim.0,
            dim_out: dim.1,
        }
    }

    fn view_params<'a>(&self, params: &'a [f32]) -> (ArrayView2<'a, f32>, ArrayView1<'a, f32>) {
        let &Self { dim_in, dim_out } = self;

        let w_range = ..params.len() - dim_out;
        let b_range = params.len() - dim_out..params.len();

        let weights = ArrayView2::from_shape((dim_in, dim_out), &params[w_range]).unwrap();
        let biases = ArrayView1::from_shape(dim_out, &params[b_range]).unwrap();

        (weights, biases)
    }

    pub fn forward(&mut self, params: &mut &[f32], x: ArrayView2<f32>) -> Array2<f32> {
        let Self { dim_in, dim_out } = *self;

        let (weights, biases) = self.view_params(params);
        let offset = (dim_in + 1) * dim_out;
        *params = &params[offset..];

        weights.dot(&x) + biases
    }

    pub fn backward(&mut self, params: &mut &[f32], d: ArrayViewMut2<f32>) -> Array2<f32> {
        let &mut Self { dim_in, dim_out } = self;

        let offset = params.len() - (dim_in + 1) * dim_out;
        let (weights, _biases) = self.view_params(&params[offset..]);
        *params = &params[offset..];

        weights.dot(&d) // FIXME
    }
}
