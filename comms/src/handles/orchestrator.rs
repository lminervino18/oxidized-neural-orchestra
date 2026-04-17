use std::io;

use tokio::io::AsyncWrite;

use crate::{share_dataset, transport::TransportLayer};

/// The handle for communicating with an `Orchestrator`.
pub struct OrchHandle<T: TransportLayer> {
    transport: T,
}

impl<T> OrchHandle<T>
where
    T: TransportLayer,
{
    /// Creates a new `OrchHandle`.
    ///
    /// # Args
    /// * `transport` - The transport layer of the communication.
    ///
    /// # Returns
    /// A new `OrchHandle` instance.
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    /// Waits to receive the dataset from the orchestrator and writes both samples
    /// and labels to the given writers.
    ///
    /// # Args
    /// * `xs` - The sink for samples.
    /// * `ys` - The sink for labels.
    /// * `xs_size` - The total size of the samples in bytes.
    /// * `ys_size` - The total size of the labels in bytes.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn pull_dataset<W>(
        &mut self,
        xs: &mut W,
        ys: &mut W,
        xs_size: usize,
        ys_size: usize,
    ) -> io::Result<()>
    where
        W: AsyncWrite + Unpin,
    {
        share_dataset::recv_dataset(xs, ys, xs_size, ys_size, &mut self.transport).await
    }
}
