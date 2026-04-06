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
    let x_size = 254 * size_of::<f32>();
    let y_size = 127 * size_of::<f32>();
    let samples: Vec<u8> = (0..x_size).map(|i| i as u8).collect();
    let labels: Vec<u8> = (0..y_size).map(|i| i as u8).collect();

    let mut recvr_x_storage = vec![];
    let mut recvr_y_storage = vec![];
    let mut sender_x_storage: &[u8] = &samples;
    let mut sender_y_storage: &[u8] = &labels;

    let send = send_dataset(
        &mut sender_x_storage,
        &mut sender_y_storage,
        chunk,
        &mut sender,
    );
    let recv = recv_dataset(
        &mut recvr_x_storage,
        &mut recvr_y_storage,
        x_size,
        y_size,
        &mut receiver,
    );

    let (send_result, recv_result) = tokio::join!(send, recv);
    send_result.unwrap();
    recv_result.unwrap();

    assert_eq!(samples, recvr_x_storage);
    assert_eq!(labels, recvr_y_storage);
}
