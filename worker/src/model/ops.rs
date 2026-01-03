//! Math kernels (placeholder).
use super::{layout::ParameterLayout, view::LinearRegressionView};

/// Computes gradients for linear regression (1D) under MSE loss:
///
/// loss = (1/n) * sum_i (yhat_i - y_i)^2
///
/// grads:
/// - dL/dw = (2/n) * sum_i (err_i * x_i)
/// - dL/db = (2/n) * sum_i (err_i)
///
/// This writes into `grads` (flat buffer) using the provided layout.
/// Requirements:
/// - weights: flat [w, b]
/// - grads: flat [dw, db] (same length as weights)
pub fn linreg_mse_grad_batch(
    layout: &ParameterLayout,
    weights: &[f32],
    grads: &mut [f32],
    xs: &[f32],
    ys: &[f32],
) {
    assert_eq!(xs.len(), ys.len(), "xs and ys must match");
    assert!(!xs.is_empty(), "batch must be non-empty");
    assert!(weights.len() >= layout.b.end, "weights too small for layout");
    assert!(grads.len() >= layout.b.end, "grads too small for layout");

    // Make sure we start from a clean accumulator for this call.
    // (Callers may already have zeroed the buffer; this is safe.)
    grads.fill(0.0);

    let n = xs.len() as f32;
    let two_over_n = 2.0 / n;

    let view = LinearRegressionView::new(weights, layout);

    let mut dw = 0.0_f32;
    let mut db = 0.0_f32;

    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let pred = view.predict(x);
        let err = pred - y;
        dw += err * x;
        db += err;
    }

    dw *= two_over_n;
    db *= two_over_n;

    grads[layout.w.start] = dw;
    grads[layout.b.start] = db;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{layout::ParameterLayout, spec::ModelSpec};

    #[test]
    fn linreg_grad_matches_expected_simple_case() {
        // Small deterministic batch
        // y = 2x + 1, but weights start at w=0, b=0
        let xs = [1.0_f32, 2.0, 3.0];
        let ys = [3.0_f32, 5.0, 7.0];

        let spec = ModelSpec::LinearRegression1D;
        let layout = ParameterLayout::new(spec);
        layout.validate(spec.num_params());

        let weights = [0.0_f32, 0.0_f32]; // w=0,b=0
        let mut grads = [0.0_f32, 0.0_f32];

        linreg_mse_grad_batch(&layout, &weights, &mut grads, &xs, &ys);

        // Compute expected:
        // preds: [0,0,0]
        // errs:  [-3,-5,-7]
        // dL/dw = (2/3)*sum(err*x)= (2/3)*(-3*1 + -5*2 + -7*3) = (2/3)*(-34)= -22.666666...
        // dL/db = (2/3)*sum(err)  = (2/3)*(-15) = -10
        let dw_expected = -22.666666_f32;
        let db_expected = -10.0_f32;

        assert!((grads[0] - dw_expected).abs() < 1e-4);
        assert!((grads[1] - db_expected).abs() < 1e-4);
    }
}
