use super::{feedforward::Feedforward, model::Model};
use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};

pub struct Mlp {
    pub activations: Vec<Array1<f32>>,
    pub weighted_sums: Vec<Array1<f32>>,
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array1<f32>>,
    pub sigmoid: fn(f32) -> f32,
    pub sigmoid_prime: fn(f32) -> f32,
}

impl Mlp {
    pub fn new(sizes: &[usize], sigmoid: fn(f32) -> f32, sigmoid_prime: fn(f32) -> f32) -> Self {
        let activations: Vec<_> = sizes
            .iter()
            .map(|s| Array1::random(*s, StandardNormal))
            .collect();
        let weights = (0..sizes.len() - 1)
            .map(|idx| Array2::random((sizes[idx + 1], sizes[idx]), StandardNormal))
            .collect();

        Self {
            weighted_sums: activations[1..].to_vec(),
            biases: activations[1..].to_vec(),
            activations,
            weights,
            sigmoid,
            sigmoid_prime,
        }
    }
}

impl Model for Mlp {}

impl Feedforward for Mlp {
    fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
        self.activations[0] = x;
        (0..self.weighted_sums.len()).for_each(|idx| {
            let z = &mut self.weighted_sums[idx];
            // let input = &mut self.activations[idx];
            // let output = &mut self.activations[idx + 1];
            let w = &self.weights[idx];
            let b = &self.biases[idx];
            let sigmoid = &(self.sigmoid);

            *z = w.dot(&self.activations[idx]) + b;
            self.activations[idx + 1] = z.mapv(sigmoid);
        });

        self.activations.last().unwrap().clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test01() {
        let _net = Mlp::new(&[2, 3, 1], |x| x, |_| 1.);
    }

    #[test]
    fn test02() {
        let mut net = Mlp::new(&[2, 3, 1], |x| x, |_| 1.);
        let x = Array1::from_vec(vec![1., 2.]);
        net.forward(x);
    }
}
