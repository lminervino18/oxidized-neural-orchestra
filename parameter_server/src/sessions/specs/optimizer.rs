pub enum OptimizerSpec {
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
    GradientDescent {
        learning_rate: f32,
    },
    GradientDescentWithMomentum {
        learning_rate: f32,
        momentum: f32,
    },
}
