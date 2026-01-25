use machine_learning::arch::{layers::Layer, loss::Mse, Sequential};
use ndarray::Array2;

fn main() {
    let xs = vec![0., 0., 0., 1., 1., 0., 1., 1.];
    let ys = vec![0., 0., 0., 1.];

    let x = Array2::from_shape_vec((xs.len() / 2, 2), xs).unwrap();
    let y = Array2::from_shape_vec((ys.len(), 1), ys).unwrap();

    let mut model = Sequential::new([
        Layer::dense((2, 2)),
        Layer::sigmoid(2, 1.0),
        Layer::dense((2, 1)),
        Layer::sigmoid(1, 1.),
    ]);
    let mut optimizer = GradientDescent::new(model, 0.1);

    let nparams = 4 + 2;
    let params = [0.].repeat(nparams);
    let mut grad = vec![0.; nparams];

    for _ in 0..100 {
        let y_pred = model.forward(&params, x.clone());
        model.backward(&params, y_pred.view(), y.view(), &Mse);
        let grad = model.take_grad(&mut grad);

        // optimize
    }

    println!("w1: {:?}", &params[..2]);
    println!("b1: {:?}", &params[2..4]);
    println!("w2: {:?}", &params[4..5]);
    println!("b2: {:?}", &params[5..]);
}
