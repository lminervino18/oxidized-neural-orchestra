use comms::{
    Float01, Serializer,
    msg::{Msg, Payload},
};
use tokio::io;

#[tokio::test]
async fn test_sparse_gradient() -> io::Result<()> {
    let (sv_stream, wk_stream) = io::duplex(1024);
    let r = Float01::new(0.4).unwrap();
    let seed = Some(0);

    let (rx, tx) = io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(rx, tx, Serializer::default());

    let (rx, tx) = io::split(wk_stream);
    let (mut wk_rx, mut wk_tx) = comms::channel(rx, tx, Serializer::new_sparse_capable(r, seed));

    // Server sends parameters to worker
    let mut params: Vec<_> = (0..5).map(|i| i as f32).collect();
    let msg = Msg::Data(Payload::Params(&mut params));
    sv_tx.send(&msg).await?;

    // Worker processes them and send the gradient
    let Msg::Data(Payload::Params(params)) = wk_rx.recv(None).await? else {
        panic!();
    };

    let mut residual = params.to_vec();
    let msg = Msg::Data(Payload::Grad(&mut residual));
    wk_tx.send(&msg).await?;

    // Server reads the new gradient.
    let mut nums_buf = vec![0.0; params.len()];
    let Msg::Data(Payload::Grad(grad)) = sv_rx.recv(nums_buf.as_mut_slice()).await? else {
        panic!();
    };

    assert_eq!(grad, [0.0, 0.0, 0.0, 3.0, 4.0]);
    Ok(())
}
