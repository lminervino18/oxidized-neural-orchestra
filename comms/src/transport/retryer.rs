use std::{io, mem, time::Duration};

use tokio::time;

use super::TransportLayer;
use crate::protocol::Msg;

/// The `Retryer` retries sending and receiving messages using exponential backoff.
pub struct Retryer<L: TransportLayer> {
    base: Duration,
    coef: u32,
    retries: usize,
    inner: L,
}

impl<L: TransportLayer> Retryer<L> {
    /// Creates a new `Retryer` transport layer.
    ///
    /// # Args
    /// * `base` - The base sleep duration.
    /// * `coef` - The coeficient to which to multiply the current sleep duration.
    /// * `retries` - The amount of retries to do per message.
    /// * `inner` - The inner transport layer stack.
    ///
    /// # Returns
    /// A new `Retryer` transport layer instance.
    pub fn new(base: Duration, coef: u32, retries: usize, inner: L) -> Self {
        Self {
            base,
            coef,
            retries,
            inner,
        }
    }

    /// Decides wheather a given `io::Error` is meant to be retried.
    ///
    /// # Args
    /// * `e` - An error in the communication.
    ///
    /// # Returns
    /// `true` if this error should be retried, `false` otherwise.
    fn is_retriable(e: &io::Error) -> bool {
        type EK = io::ErrorKind;

        match e.kind() {
            EK::ConnectionReset
            | EK::ConnectionAborted
            | EK::BrokenPipe
            | EK::TimedOut
            | EK::UnexpectedEof => true,
            _ => false,
        }
    }

    /// Given the current sleep duration, returns the next sleep duration.
    ///
    /// # Args
    /// * `sleep_dur` - The current sleep duration.
    ///
    /// # Returns
    /// The next sleep duration.
    fn next_backoff(&self, sleep_dur: Duration) -> Duration {
        self.coef * sleep_dur
    }
}

impl<L: TransportLayer> TransportLayer for Retryer<L> {
    /// Will attempt to receive a message with retries.
    ///
    /// # Returns
    /// A deserialized `Msg` or an io error if occurred.
    async fn recv(&mut self) -> io::Result<Msg<'_>> {
        let mut sleep_dur = self.base;

        for _ in 0..self.retries {
            match self.inner.recv().await {
                Ok(msg) => {
                    // SAFETY: The message's inner lifetime
                    //         outlives '1.
                    return Ok(unsafe { mem::transmute(msg) });
                }
                Err(e) if Self::is_retriable(&e) => {
                    time::sleep(sleep_dur).await;
                    sleep_dur = self.next_backoff(sleep_dur);
                }
                Err(e) => return Err(e),
            }
        }

        self.inner.recv().await
    }

    /// Will attempt to send the message with retries.
    ///
    /// # Args
    /// * `msg` - The message to send.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn send(&mut self, msg: &Msg<'_>) -> io::Result<()> {
        let mut sleep_dur = self.base;

        for _ in 0..self.retries {
            match self.inner.send(msg).await {
                Err(e) if Self::is_retriable(&e) => {
                    time::sleep(sleep_dur).await;
                    sleep_dur = self.next_backoff(sleep_dur);
                }
                other => return other,
            }
        }

        self.inner.send(msg).await
    }
}
