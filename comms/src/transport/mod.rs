mod framer;
mod layer;
mod retryer;
mod timeouter;

use std::time::Duration;

pub use framer::Framer;
pub use layer::TransportLayer;
pub use retryer::Retryer;
pub use timeouter::TimeOuter;
use tokio::io::{AsyncRead, AsyncWrite};

/// The reliable transport.
pub type Rtp<R, W> = Retryer<TimeOuter<Framer<R, W>>>;

/// The simple transport;
pub type Stp<R, W> = Framer<R, W>;

/// Builds an uninitialized reliable transport.
///
/// # Args
/// * `reader` - The reading end of the communication.
/// * `writer` - The writing end of the communication.
/// * `timeout` - The timeout duration to wait for the receival of a message.
/// * `base_retry_dur` - The base duration for the exponential backoff retryer.
/// * `retry_coef` - The coefficient to which to multiply the current wait duration.
/// * `retries` - The amount of retries till declaring a dead node.
///
/// # Returns
/// An uninitialized `Rtp` instance.
pub fn build_reliable_transport<R, W>(
    reader: R,
    writer: W,
    timeout: Duration,
    base_retry_dur: Duration,
    retry_coef: u32,
    retries: usize,
) -> Rtp<R, W>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let framer = Framer::new(reader, writer);
    let timeouter = TimeOuter::new(timeout, framer);
    let retryer = Retryer::new(base_retry_dur, retry_coef, retries, timeouter);
    retryer
}

/// Builds an uninitialized simple transport.
///
/// # Args
/// * `reader` - The reading end of the communication.
/// * `writer` - The writing end of the communication.
///
/// # Returns
/// An uninitialized `Stp` instance.
pub fn build_simple_transport<R, W>(reader: R, writer: W) -> Stp<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    Framer::new(reader, writer)
}
