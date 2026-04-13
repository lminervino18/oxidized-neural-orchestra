use comms::{
    msg::{Msg, Payload},
    Float01,
};
use tokio::io;

#[tokio::test]
async fn test_sparse_gradient() -> io::Result<()> {
    const GRAD_SIZE: usize = 16;

    let (sv_stream, wk_stream) = io::duplex(1024);
    let r = Float01::new(0.4).unwrap();
    let seed = Some(0);

    let (rx, tx) = io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::sparse_rx_channel(rx, tx, GRAD_SIZE);

    let (rx, tx) = io::split(wk_stream);
    let (mut wk_rx, mut wk_tx) = comms::sparse_tx_channel(rx, tx, r, seed);

    let mut params: Vec<_> = (0..GRAD_SIZE).map(|i| i as f32).collect();
    let msg = Msg::Data(Payload::Params(&mut params));
    sv_tx.send(&msg).await?;

    let Msg::Data(Payload::Params(params)) = wk_rx.recv().await? else {
        panic!();
    };

    let grad = params.to_vec();
    let msg = Msg::Data(Payload::Grad(&grad));
    wk_tx.send(&msg).await?;

    let Msg::Data(Payload::Grad(grad)) = sv_rx.recv().await? else {
        panic!();
    };

    assert_eq!(
        grad,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    );
    Ok(())
}
