use super::Optimizer;

#[derive(Debug)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    beta1_t: f32,
    beta2_t: f32,
    v: Box<[f32]>,
    s: Box<[f32]>,
    epsilon: f32,
}

impl Adam {
    pub fn new(len: usize, learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            beta1_t: 1.,
            beta2_t: 1.,
            v: vec![0.; len].into_boxed_slice(),
            s: vec![0.; len].into_boxed_slice(),
            epsilon,
        }
    }
}

impl Optimizer for Adam {
    fn update_weights(&mut self, grad: &[f32], weights: &mut [f32]) {
        let Self {
            learning_rate: lr,
            beta1: b1,
            beta2: b2,
            epsilon: eps,
            ..
        } = *self;

        self.beta1_t *= b1;
        self.beta2_t *= b2;

        let bc1 = 1. - self.beta1_t;
        let bc2 = 1. - self.beta2_t;
        let step_size = lr * (bc2.sqrt() / bc1);

        weights
            .iter_mut()
            .zip(grad)
            .zip(self.v.iter_mut())
            .zip(self.s.iter_mut())
            .for_each(|(((w, g), v), s)| {
                *v = b1 * *v + (1. - b1) * g;
                *s = b2 * *s + (1. - b2) * g.powi(2);
                *w -= step_size * *v / (s.sqrt() + eps);
            });
    }
}
