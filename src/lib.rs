use ndarray::prelude::*;

struct NeuralNet {
    sizes: Vec<usize>,
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
}

impl NeuralNet {
    fn new(sizes: Vec<usize>) -> Self {
        let weights = (0..sizes.len() - 1)
            .map(|idx| Array::<f32, Ix2>::zeros((sizes[idx + 1], sizes[idx])))
            .collect();

        let biases = sizes
            .iter()
            .map(|size| Array::<f32, Ix1>::zeros(*size))
            .collect();

        Self {
            sizes,
            weights,
            biases,
        }
    }

    fn forward(&self, x: Array<f32, Ix1>) -> Array1<f32> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test01() {
        let _net = NeuralNet::new(vec![2, 3, 1]);
    }
}
