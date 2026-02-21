mod align;
mod deserialize;
pub mod msg;
mod receiver;
mod sender;
mod serialize;
pub mod specs;

use tokio::io::{AsyncRead, AsyncWrite};

pub use align::{Align1, Align4};
pub use deserialize::Deserialize;
pub use receiver::OnoReceiver;
pub use sender::OnoSender;
pub use serialize::Serialize;

type LenType = u64;
const LEN_TYPE_SIZE: usize = size_of::<LenType>();

/// Creates both `OnoReceiver` and `OnoSender` network channel parts.
///
/// Given a writer and reader creates and returns both ends of the communication.
///
/// # Arguments
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
    (OnoReceiver::new(rx), OnoSender::new(tx))
}
