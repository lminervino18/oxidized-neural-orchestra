//! Random, playful file names for auto-saved models, e.g. `nice_gradient`.

use std::time::{SystemTime, UNIX_EPOCH};

/// Short, friendly adjectives.
const ADJECTIVES: [&str; 50] = [
    "nice", "cute", "super", "tiny", "mega", "bold", "calm", "cool", "warm", "wild", "fuzzy",
    "shiny", "silky", "brave", "chill", "crisp", "dizzy", "eager", "fancy", "giant", "happy",
    "jolly", "lucky", "merry", "noble", "plush", "quick", "rapid", "sassy", "snug", "spicy",
    "swift", "witty", "zesty", "bouncy", "breezy", "cheery", "cosmic", "dreamy", "gentle", "glowy",
    "groovy", "jazzy", "lively", "mighty", "peppy", "quirky", "sleepy", "sneaky", "sunny",
];

/// Deep-learning / ML flavored terms.
const TERMS: [&str; 50] = [
    "transformer", "hopfield", "gradient", "backprop", "tensor", "neuron", "softmax", "sigmoid",
    "relu", "dropout", "epoch", "weight", "bias", "kernel", "embedding", "attention", "encoder",
    "decoder", "perceptron", "convolution", "pooling", "adam", "sgd", "momentum", "entropy",
    "logit", "activation", "batchnorm", "layernorm", "autograd", "manifold", "eigenvector",
    "jacobian", "hessian", "optimizer", "regressor", "classifier", "boltzmann", "autoencoder",
    "lstm", "gru", "resnet", "bert", "gan", "diffusion", "latent", "feature", "sampler", "learner",
    "network",
];

/// Returns a random `<adjective>_<term>.safetensors` file name.
///
/// The two words are drawn from a time-seeded xorshift, so no extra dependency
/// is needed just to pick a name.
pub fn random_model_name() -> String {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);

    let mut x = seed | 1;
    let adjective = ADJECTIVES[(next(&mut x) as usize) % ADJECTIVES.len()];
    let term = TERMS[(next(&mut x) as usize) % TERMS.len()];

    format!("{adjective}_{term}.safetensors")
}

/// Advances an xorshift64 state and returns the new value.
fn next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
