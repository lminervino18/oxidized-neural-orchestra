mod test_sequential_conv_dense;
mod test_sequential_dense;

use rand::Rng;

fn gen_params_grads(
    server_sizes: &[usize],
    rng: &mut impl Rng,
) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    server_sizes
        .iter()
        .map(|&size| {
            (
                (0..size).map(|_| rng.random_range(-0.5..0.5)).collect(),
                vec![0.; size],
                vec![0.; size],
            )
        })
        .collect()
}
