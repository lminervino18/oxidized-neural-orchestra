use ndarray::prelude::*;

struct NeuralNet {
    sizes: Vec<usize>,
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
    activation: fn(f32) -> f32,
}

fn sigmoid(z: f32) -> f32 {
    1. / (1. - std::f32::consts::E.powf(z))
}

impl NeuralNet {
    fn new(sizes: Vec<usize>, activation: fn(f32) -> f32) -> Self {
        let weights = (0..sizes.len() - 1)
            .map(|idx| Array2::<f32>::zeros((sizes[idx + 1], sizes[idx])))
            .collect();

        let biases = sizes[1..]
            .iter()
            .map(|size| Array1::<f32>::zeros(*size))
            .collect();

        Self {
            sizes,
            weights,
            biases,
            activation,
        }
    }

    fn forward(&self, x: Array1<f32>) -> Array1<f32> {
        let mut y_pred = x;
        self.weights.iter().zip(&self.biases).for_each(|(w, b)| {
            y_pred = (w.dot(&y_pred) + b).map(|z| sigmoid(*z));
        });

        y_pred
    }

    fn train(&mut self, x_train: Vec<Array1<f32>>, y_train: Vec<Array1<f32>>) {
        x_train.iter().zip(y_train).for_each(|(x, y)| {
            let y_pred = self.forward(x.clone() /* grave...*/);
            let err = l2_distance(y_pred.view(), y.view());

            // grad
            // update
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test01() {
        let _net = NeuralNet::new(vec![2, 3, 1], sigmoid);
    }

    #[test]
    fn test02() {
        let net = NeuralNet::new(vec![2, 3, 1], sigmoid);
        let x = Array1::<f32>::from_vec(vec![1., 2.]);
        net.forward(x);
    }
}
