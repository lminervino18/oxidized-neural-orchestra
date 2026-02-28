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

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn forward(&mut self, params: &[f32], x: ArrayView2<f32>) -> Result<ArrayView2<'_, f32>> {
        let (w, b) = self.view_params(params)?;
        let shape = (x.nrows(), self.dim.1);

        self.z = self.z.reshape_inplace(shape);
        linalg::general_mat_mul(1.0, &x, &w, 0.0, &mut self.z);
        self.z += &b;

        // TODO: See if this `to_owned` call can be removed.
        self.x = x.to_owned();

        let Some(ref act_fn) = self.act_fn else {
            return Ok(self.z.view());
        };

        self.a = self.a.reshape_inplace(shape);
        self.a.zip_mut_with(&self.z, |a, &z| *a = act_fn.f(z));
        Ok(self.a.view())
    }

    pub fn backward(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        mut d: ArrayViewMut2<f32>,
    ) -> Result<ArrayViewMut2<'_, f32>> {
        if let Some(act_fn) = &self.act_fn {
            d.zip_mut_with(&self.z, |d, &z| *d *= act_fn.df(z));
        }

        let (mut dw, mut db) = self.view_grad(grad)?;
        linalg::general_mat_mul(1.0, &self.x.t(), &d, 0.0, &mut dw);
        db.view_mut().assign(&d.sum_axis(Axis(0)));

        let (w, _) = self.view_params(params)?;
        self.d = self.d.reshape_inplace((d.nrows(), w.nrows()));
        linalg::general_mat_mul(1.0, &d, &w.t(), 0.0, &mut self.d);

        Ok(self.d.view_mut())
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
        if grad.len() != self.size {
            return Err(MlErr::SizeMismatch {
                what: "grad",
                got: grad.len(),
                expected: self.size,
            });
        }

        let w_size = self.size - self.dim.1;

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
        let w_size = self.size - self.dim.1;
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
        let mut dense = Dense::new((3, 2), None);
        /*      [1 2
         *  W =  3 4 , B = [7 8]
         *       5 6]
         ***/
        let params = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        /* x =  [9 10 11]
         ***/
        let x = ArrayView2::from_shape((1, 3), &[9.0, 10.0, 11.0]).unwrap();

        let expected = ArrayView2::from_shape(
            (1, 2),
            &[
                1. * 9. + 3. * 10. + 11. * 5. + 7.,
                2. * 9. + 4. * 10. + 11. * 6. + 8.,
            ],
        )
        .unwrap();

        let y = dense.forward(&params, x).unwrap();

        assert_eq!(y, expected);
    }
}
