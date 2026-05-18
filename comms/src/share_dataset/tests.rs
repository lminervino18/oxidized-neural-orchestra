#![cfg(test)]

use std::env;

use tokio::io::{self, duplex};

use super::{recv_dataset, send_dataset};
use crate::{share_dataset, transport::Framer};

#[tokio::test]
async fn test_share_dataset() {
    unsafe { env::set_var("RUST_BACKTRACE", "1") };
    const SIZE: usize = 2 << 13;

    let (rx, tx) = duplex(SIZE);
    let (rx, _) = io::split(rx);
    let (_, tx) = io::split(tx);
    let mut transport = Framer::new(rx, tx);

    let chunk_size = 4;
    let x_size = 254;
    let y_size = 127;
    let samples: Vec<_> = (0..x_size).map(|i| i as f32).collect();
    let labels: Vec<_> = (0..y_size).map(|i| i as f32).collect();

    let mut rx_x_storage = vec![];
    let mut rx_y_storage = vec![];
    let mut tx_x_cursor = share_dataset::get_dataset_cursor(&samples);
    let mut tx_y_cursor = share_dataset::get_dataset_cursor(&labels);

    send_dataset(
        &mut tx_x_cursor,
        &mut tx_y_cursor,
        x_size,
        y_size,
        chunk_size,
        &mut transport,
    )
    .await
    .unwrap();

    recv_dataset(&mut rx_x_storage, &mut rx_y_storage, &mut transport)
        .await
        .unwrap();

    assert_eq!(samples, rx_x_storage);
    assert_eq!(labels, rx_y_storage);
}
