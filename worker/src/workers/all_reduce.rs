use std::io;

use comms::{OnoReceiver, OnoSender};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::middlewares::all_reduce::AllReduceMiddleware;

/// The worker implementation for the all-reduce distributed algorithm.
pub struct AllReduceWorker {
    _worker_addr: String,
    _worker_addrs: Vec<String>,
}

impl AllReduceWorker {
    /// Creates a new `AllReduceWorker`.
    ///
    /// # Returns
    /// A new `AllReduceWorker` instance.
    pub fn new(worker_addr: String, worker_addrs: Vec<String>) -> Self {
        Self {
            _worker_addr: worker_addr,
            _worker_addrs: worker_addrs,
        }
    }

    /// Runs the worker using the all-reduce distributed algorithm.
    ///
    /// # Args
    /// * `rx` - The receiving end of the communication between the worker and the orchestrator.
    /// * `tx` - The sending end of the communication between the worker and the orchestrator.
    /// * `middleware` - The communication manager between this worker and the all-reduce peers.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn run<R, W>(
        self,
        _rx: OnoReceiver<R>,
        _tx: OnoSender<W>,
        _middleware: AllReduceMiddleware<R, W>,
    ) -> io::Result<()>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        // TODO: Use the local ML state together with the ring topology to implement
        //       the all-reduce worker execution.
        Err(io::Error::other("all-reduce worker is not implemented yet"))
    }
}
