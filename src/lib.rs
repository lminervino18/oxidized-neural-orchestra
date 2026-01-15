use ndarray::prelude::*;

struct NeuralNet {
    sizes: Vec<usize>,
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
    activation: fn(f32) -> f32,
}

fn l2_distance(v: ArrayView1<f32>, w: ArrayView1<f32>) -> f32 {
    v.iter()
        .zip(w)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
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

    // recorrer el dataset -> backprop e ir sumando el grad resultante al grad total -> moverse en
    // la dirección contraria al gradiente resultante escalado con el learning rate (eta)
    fn train(&mut self, x_train: Vec<Array1<f32>>, y_train: Vec<Array1<f32>>, eta: f32) {
        let mut grad_w: Vec<_> = self
            .weights
            .iter()
            .map(|w| Array2::<f32>::zeros(w.dim()))
            .collect();

        let mut grad_b: Vec<_> = self
            .biases
            .iter()
            .map(|b| Array1::<f32>::zeros(b.dim()))
            .collect();

        x_train.iter().zip(y_train).for_each(|(x, y)| {
            let delta_grad_w;
            let delta_grad_b;

            // TODO: backprop

            grad_w.iter_mut().for_each(|gw| *gw += delta_grad_w);
            grad_b.iter_mut().for_each(|gb| *gb += delta_grad_b);
        });

        self.weights
            .iter_mut()
            .zip(grad_w)
            .for_each(|(w, gw)| w.scaled_add(-eta, &gw));

        self.biases
            .iter_mut()
            .zip(grad_b)
            .for_each(|(b, gb)| w.scaled_add(-eta, &gb));
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

    fn test03() {
        let a = Array1::<f32>::zeros(10);
        let vec = vec![a];
        let b = Array1::<f32>::zeros(vec.first().unwrap().dim());
    }
}
