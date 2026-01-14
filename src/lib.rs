use ndarray::prelude::*;

struct NeuralNet {
    sizes: Vec<usize>,
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
}

impl NeuralNet {
    fn new(sizes: Vec<usize>) -> Self {
        let weights = (0..sizes.len() - 1)
            .map(|idx| Array2::<f32>::zeros((sizes[idx + 1], sizes[idx])))
            .collect();

        let biases = sizes
            .iter()
            .map(|size| Array1::<f32>::zeros(*size))
            .collect();

        Self {
            sizes,
            weights,
            biases,
        }
    }

    fn forward(&self, x: Array1<f32>) -> Array1<f32> {
        let mut y_pred = x;
        self.weights.iter().zip(&self.biases).for_each(|(w, b)| {
            y_pred = w.dot(&y_pred) + b;
        });

        y_pred
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test01() {
        let _net = NeuralNet::new(vec![2, 3, 1]);
    }

    #[test]
    fn test02() {
        let net = NeuralNet::new(vec![2, 3, 1]);
        let x = Array1::<f32>::from_vec(vec![1., 2.]);
        net.forward(x);
    }
}
