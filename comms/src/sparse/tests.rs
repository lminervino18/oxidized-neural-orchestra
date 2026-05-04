#![cfg(test)]

use rand::{SeedableRng, rngs::StdRng};
use tokio::io;

use crate::{
    floats::Float01,
    protocol::{Msg, Payload},
    sparse,
    transport::{Framer, TransportLayer},
};

#[tokio::test]
async fn test_sparse_gradient() -> io::Result<()> {
    const GRAD_SIZE: usize = 16;

    let (sv_stream, wk_stream) = io::duplex(1024);
    let r = Float01::new(0.4).unwrap();
    let mut rng = StdRng::seed_from_u64(0);

    let (rx, tx) = io::split(sv_stream);
    let mut sv_transport = Framer::new(rx, tx);

    let (rx, tx) = io::split(wk_stream);
    let mut wk_transport = Framer::new(rx, tx);

    let mut params: Vec<_> = (0..GRAD_SIZE).map(|i| i as f32).collect();
    let msg = Msg::Data(Payload::Params(&mut params));
    sv_transport.send(&msg).await?;

    let Msg::Data(Payload::Params(params)) = wk_transport.recv().await? else {
        panic!("Didn't receive parameters");
    };

    let mut grad = params.to_vec();
    let mut ser_buf = Vec::new();

    let threshold = sparse::calculate_threshold(&grad, r, &mut rng);
    sparse::grad_drop_into(&mut ser_buf, &grad, threshold);

    let msg = Msg::Data(Payload::SparseGrad(&ser_buf));
    wk_transport.send(&msg).await?;

    let Msg::Data(Payload::SparseGrad(sparse)) = sv_transport.recv().await? else {
        panic!("Didn't receive a sparse gradient");
    };

    grad.fill(0.0);
    sparse::grad_lift_into(&mut grad, sparse).map_err(io::Error::other)?;

    let expected = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    assert_eq!(grad, expected);
    Ok(())
}
