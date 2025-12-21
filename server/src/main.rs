mod executors;
mod optimization;
mod parameter_server;

fn main() {
    use optimization::Optimizer;
    use parameter_server::PSServer;
    use rayon::prelude::*;

    #[derive(Debug)]
    struct TestOptimizer {}

    impl Optimizer for TestOptimizer {
        #![allow(unused_variables)]
        fn update_weights(&mut self, weights: &mut [f32], gradient: &[f32]) {
            weights
                .par_iter_mut()
                .zip(gradient.par_iter())
                .for_each(|(w, g)| {
                    *w -= g;
                });
        }
    }

    let mut ps = PSServer::new(3, TestOptimizer {});
    let pc = ps.client_handle();

    let gradient = [1., 2., 3.];
    pc.accumulate(&gradient);
    println!("{ps:#?}");

    ps.update_weights();
    println!("{ps:#?}");

    let gradient = [3., 2., 1.];
    pc.accumulate(&gradient);
    println!("{ps:#?}");

    ps.update_weights();
    println!("{ps:#?}");
}
