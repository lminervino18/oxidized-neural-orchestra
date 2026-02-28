use ndarray::{linalg, prelude::*};

use super::InplaceReshape;
use crate::{MlErr, Result, arch::activations::ActFn};

/// Optimizations:
///   1. Find a way to not copy `x` in each `Dense::forward` call.
///   2. Find a way to sum up the rows of `d` in `Dense::backward` in parallel to write them to `b`.
///   3. Maybe make the a = f(z) computation parallel.
///   4. Maybe make the d *= df(z) computation parallel.
#[derive(Clone)]
pub struct Dense {
    dim: (usize, usize),
    nparams: usize,

    // Forward metadata
    input: Array2<f32>,
    w_sums: Array2<f32>,

    // Backward metadata
    delta: Array2<f32>,
}

trait MultiplyInplace {
    fn multiply_into(&mut self, matrix: ArrayView2<f32>, other: ArrayView2<f32>);
}

impl MultiplyInplace for Array2<f32> {
    fn multiply_into(&mut self, matrix: ArrayView2<f32>, other: ArrayView2<f32>) {
        linalg::general_mat_mul(1.0, &matrix, &other, 0.0, self);
    }
}

impl MultiplyInplace for ArrayViewMut2<'_, f32> {
    fn multiply_into(&mut self, matrix: ArrayView2<f32>, other: ArrayView2<f32>) {
        linalg::general_mat_mul(1.0, &matrix, &other, 0.0, self);
    }
}

impl Dense {
    pub fn new(dim: (usize, usize)) -> Self {
        let zeros = Array2::zeros((1, 1));

        Self {
            dim,
            nparams: (dim.0 + 1) * dim.1,
            input: zeros.clone(),
            w_sums: zeros.clone(),
            delta: zeros,
        }
    }

    pub fn size(&self) -> usize {
        self.nparams
    }

    pub fn forward(&mut self, params: &[f32], x: ArrayView2<f32>) -> Result<ArrayView2<'_, f32>> {
        let (w, b) = self.view_params(params)?;
        let shape = (x.nrows(), self.dim.1);

        self.w_sums.reshape_inplace(shape);

        self.w_sums.multiply_into(x, w);
        self.w_sums += &b;

        // TODO: See if this `to_owned` call can be removed.
        self.input = x.to_owned();

        Ok(self.w_sums.view())
    }

    pub fn backward(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        d: ArrayViewMut2<f32>,
    ) -> Result<ArrayViewMut2<'_, f32>> {
        let (mut dw, mut db) = self.view_grad(grad)?;
        linalg::general_mat_mul(1.0, &self.input.t(), &d, 0.0, &mut dw);
        db.view_mut().assign(&d.sum_axis(Axis(0)));

        let (w, _) = self.view_params(params)?;
        self.delta.reshape_inplace((d.nrows(), w.nrows()));
        linalg::general_mat_mul(1.0, &d, &w.t(), 0.0, &mut self.delta);

        Ok(self.delta.view_mut())
    }

    /// Gives a view of the raw gradient slice as the delta weights and delta biases of this layer.
    ///
    /// # Arguments
    /// * `grad` - A gradient slice.
    ///
    /// # Returns
    /// A tuple containing the delta weights and delta biases or an error if there's
    /// a mismatch between the size of the gradient and the size of the layer.
    fn view_grad<'a>(
        &self,
        grad: &'a mut [f32],
    ) -> Result<(ArrayViewMut2<'a, f32>, ArrayViewMut1<'a, f32>)> {
        if grad.len() != self.nparams {
            return Err(MlErr::SizeMismatch {
                what: "grad",
                got: grad.len(),
                expected: self.nparams,
            });
        }

        let w_size = self.nparams - self.dim.1;

        // SAFETY: The if condition above checks that the size of the
        //         gradient is exactly the size of the layer.
        let (dw_raw, db_raw) = grad.split_at_mut(w_size);
        let dw = ArrayViewMut2::from_shape(self.dim, dw_raw).unwrap();
        let db = ArrayViewMut1::from_shape(self.dim.1, db_raw).unwrap();

        Ok((dw, db))
    }

    /// Gives a view of the raw parameter slice as the weights and biases of this layer.
    ///
    /// # Arguments
    /// * `params` - A slice of parameters.
    ///
    /// # Returns
    /// A tuple containing the weights and biases or an error if there's a mismatch
    /// between the size of the gradient and the size of the layer.
    fn view_params<'a>(
        &self,
        params: &'a [f32],
    ) -> Result<(ArrayView2<'a, f32>, ArrayView1<'a, f32>)> {
        let w_size = self.nparams - self.dim.1;
        let weights = ArrayView2::from_shape(self.dim, &params[..w_size]).unwrap();
        let biases = ArrayView1::from_shape(self.dim.1, &params[w_size..]).unwrap();
        Ok((weights, biases))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_3by2_forward() {
        let mut dense = Dense::new((3, 2));
        /*      [1 2
         *  W =  3 4 , B = [7 8]
         *       5 6]
         ***/
        let params = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        /* x =  [9 10 11]
         ***/
        let x = ArrayView2::from_shape((1, 3), &[9.0, 10.0, 11.0]).unwrap();

        /* x * W + b
         ***/
        let expected = ArrayView2::from_shape(
            (1, 2),
            &[
                1. * 9. + 3. * 10. + 11. * 5. + 7.,
                2. * 9. + 4. * 10. + 11. * 6. + 8.,
            ],
        )
        .unwrap();

        let y_pred = dense.forward(&params, x).unwrap();

        assert_eq!(y_pred, expected);
    }

    #[test]
    fn test_dense_3by2_backward() {
        use std::f32;
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

        let sigmoid = |z: f32| 1.0 / (1.0 + (-z).exp());
        let sigmoid_prime = |z: f32| sigmoid(z) * (1.0 - sigmoid(z));

        let mut dense = Dense::new((3, 2));
        /*      [.1 .2
         *  W =  .3 .4 , B = [.7 .8]
         *       .5 .6]
         ***/
        let params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut grad = vec![0.0; params.len()];

        /* x =  [.9 .10 .11]
         ***/
        let x =
            ArrayView2::from_shape((1, 3), &[0.9, 0.10, 0.11]).expect("failed building x test var");
        let y_pred = dense.forward(&params, x).expect("dense failed to forward");

        /* d = sigmoid'(y_pred) ~= [.208 .189]
         ***/
        let mut d = y_pred.mapv(sigmoid_prime);
        dbg!(&d);

        let (w, _) = dense.view_params(&params).expect("failed at view_params");

        let expected_bp = d.dot(&w.t());
        let expected_db = d.sum_axis(Axis(0));
        let expected_dw = x.t().dot(&d);

        let mut dense2 = dense.clone();

        let bp = dense2
            .backward(&params, &mut grad, d.view_mut())
            .expect("dense failed to backward");
        let (dw, db) = dense.view_params(&grad).unwrap();

        assert_eq!(bp, expected_bp);
        assert_eq!(db, expected_db);
        assert_eq!(dw, expected_dw);
    }
}
