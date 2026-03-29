mod deserializer;
pub mod msg;
mod receiver;
mod sender;
mod serializer;
mod share_dataset;
mod sparse;
pub mod specs;

use tokio::io::{AsyncRead, AsyncWrite};

pub use deserializer::Deserializer;
pub use receiver::OnoReceiver;
pub use sender::OnoSender;
pub use serializer::Serializer;
pub use share_dataset::recv_dataset;
pub use share_dataset::send_dataset;
pub use sparse::Float01;

type LenType = u64;
const LEN_TYPE_SIZE: usize = size_of::<LenType>();

/// Creates both `OnoReceiver` and `OnoSender` network channel parts.
///
/// Given a writer and reader creates and returns both ends of the communication.
///
/// # Args
/// * `rx` - An async readable.
/// * `tx` - An async writable.
///
/// # Returns
/// A communication stream in the form of an ono receiver and sender.
pub fn channel<R, W>(rx: R, tx: W) -> (OnoReceiver<R>, OnoSender<W>)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let deserializer = Deserializer::new();
    let serializer = Serializer::new();

    (
        OnoReceiver::new(rx, deserializer),
        OnoSender::new(tx, serializer),
    )
}

/// Creates both `OnoReceiver` and `OnoSender` network channel parts.
///
/// Given a writer and reader creates and returns both ends of the communication.
///
/// This constructor enables the sparse gradient compression capability as a sender.
///
/// # Args
/// * `rx` - An async readable.
/// * `tx` - An async writable.
/// * `r` - The ratio of compression for calculating the threshold value.
/// * `seed` - An optional seed to initialize the sampler's rng for the threshold calculation.
///
/// # Returns
/// A communication stream in the form of an ono receiver and sender.
pub fn sparse_tx_channel<R, W, S>(
    rx: R,
    tx: W,
    r: Float01,
    seed: S,
) -> (OnoReceiver<R>, OnoSender<W>)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    S: Into<Option<u64>>,
{
    let serializer = Serializer::new_sparse_capable(r, seed.into());

    (
        OnoReceiver::new(rx, Deserializer::new()),
        OnoSender::new(tx, serializer),
    )
}

/// Creates both `OnoReceiver` and `OnoSender` network channel parts.
///
/// Given a writer and reader creates and returns both ends of the communication.
///
/// This constructor enables the sparse gradient compression capability as a receiver.
///
/// # Args
/// * `rx` - An async readable.
/// * `tx` - An async writable.
/// * `grad_size` - The total size of the uncompressed gradient.
///
/// # Returns
/// A communication stream in the form of an ono receiver and sender.
pub fn sparse_rx_channel<R, W>(rx: R, tx: W, grad_size: usize) -> (OnoReceiver<R>, OnoSender<W>)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let deserializer = Deserializer::new_with_size(grad_size);

    (
        OnoReceiver::new(rx, deserializer),
        OnoSender::new(tx, Serializer::new()),
    )
}
