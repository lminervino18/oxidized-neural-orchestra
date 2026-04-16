use std::io;

use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{
    protocol::{Msg, Payload},
    transport::TransportLayer,
};

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
        self.recv_datachunks(xs, xs_size).await?;
        self.recv_datachunks(ys, ys_size).await?;
        Ok(())
    }

    /// Receives a dataset chunk through the transport layer and writes it's data into
    /// the given writer.
    ///
    /// # Args
    /// * `writer` - The sink for the dataset bytes.
    /// * `size` - The amount of bytes to read.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn recv_datachunks<W>(&mut self, writer: &mut W, size: usize) -> io::Result<()>
    where
        W: AsyncWrite + Unpin,
    {
        let mut received = 0;

        while received < size {
            let msg = self.transport.recv().await?;

            let Msg::Data(Payload::Datachunk(chunk)) = msg else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Expected Datachunk, got: {msg:?}"),
                ));
            };

            let bytes = bytemuck::cast_slice(chunk);
            writer.write_all(bytes).await?;
            received += bytes.len();
        }

        writer.flush().await
    }
}
