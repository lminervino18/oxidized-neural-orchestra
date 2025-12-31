mod deserialize;
mod receiver;
mod sender;
mod serialize;

use tokio::io::{AsyncRead, AsyncWrite};

pub use deserialize::Deserialize;
pub use receiver::OnoReceiver;
pub use sender::OnoSender;
pub use serialize::Serialize;

/// Creates both `OnoReceiver` and `OnoSender` network channel parts.
///
/// Given a writer and reader creates and returns both ends of the communication.
pub fn channel<R, W>(rx: R, tx: W, max_frame_size: usize) -> (OnoReceiver<R>, OnoSender<W>)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    (OnoReceiver::new(rx, max_frame_size), OnoSender::new(tx))
}
