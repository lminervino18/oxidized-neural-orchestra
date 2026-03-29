#![cfg(test)]

use tokio::io::{self, duplex};

use crate::{channel, recv_dataset::recv_dataset, send_dataset::send_dataset};

#[tokio::test]
async fn test_share_dataset() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
    const SIZE: usize = 128;

    let (rx, tx) = duplex(SIZE);
    let (rx, _) = io::split(rx);
    let (_, tx) = io::split(tx);
    let (mut receiver, mut sender) = channel(rx, tx);

    let chunk = 4;
    let size = 127 * size_of::<f32>() as u64;
    let dataset: Vec<u8> = (0..size).map(|i| i as u8).collect();

    let mut recvr_storage = vec![];
    let mut sender_storage: &[u8] = &dataset;

    let send = send_dataset(&mut sender_storage, chunk, &mut sender);
    let recv = recv_dataset(&mut recvr_storage, size, &mut receiver);

    let (send_result, recv_result) = tokio::join!(send, recv);
    send_result.unwrap();
    recv_result.unwrap();

    dbg!(&dataset);
    dbg!(&recvr_storage);
    assert_eq!(dataset, recvr_storage);
}
