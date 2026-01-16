use super::{feedforward::Feedforward, model::Model};
use ndarray::{Array1, Array2};

pub struct NeuralNet {
    activations: Vec<Array1<f32>>,
    weighted_sums: Vec<Array1<f32>>,
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
    activation: fn(f32) -> f32,
}

impl NeuralNet {
    fn new(sizes: &[usize], activation: fn(f32) -> f32) -> Self {
        let activations: Vec<_> = sizes.iter().map(|s| Array1::zeros(*s)).collect();
        let weights = (0..sizes.len() - 1)
            .map(|idx| Array2::zeros((sizes[idx + 1], sizes[idx])))
            .collect();

        Self {
            weighted_sums: activations[1..].to_vec(),
            biases: activations[1..].to_vec(),
            activations,
            weights,
            activation,
        }
    }
}

impl Model for NeuralNet {}

impl Feedforward for NeuralNet {
    fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
        let mut aux = x;
        self.weighted_sums
            .iter_mut()
            .zip(&self.weights)
            .zip(&self.biases)
            .for_each(|((z, w), b)| {
                // TODO: avoid cloning
                *z = w.dot(&aux) + b;
                aux = z.clone().mapv_into(|x| (self.activation)(x));
            });

        aux
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test01() {
        let _net = NeuralNet::new(&[2, 3, 1], |x| x);
    }

    #[test]
    fn test02() {
        let mut net = NeuralNet::new(&[2, 3, 1], |x| x);
        let x = Array1::from_vec(vec![1., 2.]);
        net.forward(x);
    }
}
