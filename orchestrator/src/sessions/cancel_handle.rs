use tokio::sync::mpsc::{self, Receiver, Sender};

/// A handle that lets any caller request an early stop of an ongoing training session.
pub struct CancelHandle(Sender<()>);

impl CancelHandle {
    /// Creates a matched `(CancelHandle, Receiver)` pair.
    ///
    /// The caller retains the `CancelHandle` and passes the `Receiver` to
    /// `Session::event_listener`. Calling `stop()` on the handle signals
    /// the session to stop at the next epoch boundary.
    pub fn pair() -> (Self, Receiver<()>) {
        let (tx, rx) = mpsc::channel(1);
        (Self(tx), rx)
    }

    /// Sends a cancel signal to the receiver.
    pub fn stop(&self) {
        let _ = self.0.try_send(());
    }
}
