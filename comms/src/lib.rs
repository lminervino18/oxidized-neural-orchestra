mod deserialize;
mod receiver;
mod sender;
mod serialize;

use tokio::io::{AsyncRead, AsyncWrite};

pub(crate) use deserialize::Deserialize;
use receiver::OnoReceiver;
use sender::OnoSender;
pub(crate) use serialize::Serialize;

type LenType = u64;
const LEN_TYPE_SIZE: usize = size_of::<LenType>();

/// Creates both `OnoReceiver` and `OnoSender` network channel parts.
///
/// Given a writer and reader creates and returns both ends of the communication.
pub fn channel<R, W>(rx: R, tx: W) -> (OnoReceiver<R>, OnoSender<W>)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    (OnoReceiver::new(rx), OnoSender::new(tx))
}
