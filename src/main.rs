use ndarray::prelude::*;

struct Perceptron {
    w: Array1<f32>,
    b: f32,
}

impl Perceptron {
    pub fn new(n: usize) -> Self {
        Self {
            w: Array::zeros(n),
            b: 0.,
        }
    }

    pub fn forward(&self, x: ArrayView1<f32>) -> f32 {
        let h = x.dot(&self.w) + self.b;
        if h >= 0. { 1. } else { -1. }
    }

    pub fn train(&mut self, x: ArrayView2<f32>, y: ArrayView1<f32>) {
        let n = x.len_of(Axis(0));

        while self.mse(x, y) > 0. {
            for i in 0..n {
                let xi = x.index_axis(Axis(0), i);
                let dyz = y[i] - self.forward(xi);
                self.w = &self.w + &xi * dyz;
                self.b += &dyz;
            }
        }
    }

    pub fn mse(&self, x: ArrayView2<f32>, y: ArrayView1<f32>) -> f32 {
        let n = x.len_of(Axis(0));
        let mut total_error = 0.;

        for i in 0..n {
            let xi = x.index_axis(Axis(0), i);
            let yi = y.get(i).expect("len(x) > len(y)");
            let prediction = self.forward(xi);
            total_error = (yi - prediction).powi(2) + total_error;
        }

        total_error / n as f32
    }
}

fn main() {
    let mut p = Perceptron::new(2);
    let x = array![[1., 1.], [1., -1.], [-1., 1.], [-1., -1.],];
    let y = array![1., -1., -1., -1.];
    p.train(x.view(), y.view());

    println!("1, 1: {}", p.forward(x.index_axis(Axis(0), 0)));
    println!("1, -1: {}", p.forward(x.index_axis(Axis(0), 1)));
    println!("-1, 1: {}", p.forward(x.index_axis(Axis(0), 2)));
    println!("-1, -1: {}", p.forward(x.index_axis(Axis(0), 3)));
}
