use std::io;

use tokio::io::AsyncRead;

use crate::{
    protocol::{Command, Msg, Payload},
    sparse,
    transport::TransportLayer,
    utils,
};

/// The handle for communicating with a `Worker`.
pub struct WorkerHandle<T> {
    id: usize,
    transport: T,
    grad: Vec<f32>,
}

/// A notified worker event.
pub enum WorkerEvent<'a> {
    Grad(&'a [f32]),
    Disconnect,
}

impl<T: TransportLayer> WorkerHandle<T> {
    /// Creates a new `WorkerHandle`.
    ///
    /// # Args
    /// * `id` - The id number of the worker.
    /// * `transport` - The transport layer of the communication.
    ///
    /// # Returns
    /// A new `WorkerHandle` instance.
    pub fn new(id: usize, transport: T) -> Self {
        Self {
            id,
            transport,
            grad: Vec::new(),
        }
    }

    /// Blocks until receiving an event from a worker.
    ///
    /// # Returns
    /// A `WorkerEvent` message or an io error if occurred.
    pub async fn recv_event(&mut self) -> io::Result<WorkerEvent> {
        self.grad.fill(0.0);

        let response = match self.transport.recv().await? {
            Msg::Data(Payload::Grad(grad)) => {
                let additional = self.grad.capacity().saturating_sub(grad.len());
                self.grad.reserve(additional);

                // SAFETY: The new uninitialized bytes will be overwritten right
                //         after with the uncompressed 32 bit gradient values.
                unsafe { self.grad.set_len(grad.len()) };

                for (r, g) in self.grad.iter_mut().zip(grad) {
                    *r = g.to_f32();
                }

                WorkerEvent::Grad(&self.grad)
            }
            Msg::Data(Payload::SparseGrad(grad)) => {
                sparse::grad_lift_into(&mut self.grad, grad).map_err(io::Error::other)?;
                WorkerEvent::Grad(&self.grad)
            }
            Msg::Control(Command::Disconnect) => WorkerEvent::Disconnect,
            msg => {
                let text = format!("Expected grads from worker {}, got: {msg:?}", self.id);
                return Err(io::Error::other(text));
            }
        };

        Ok(response)
    }

    /// Pushes the latest state of the parameters to the worker.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_params(&mut self, params: &mut [f32]) -> io::Result<()> {
        let msg = Msg::Data(Payload::Params(params));
        self.transport.send(&msg).await
    }

    /// Disconnects the worker.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Disconnect);
        self.transport.send(&msg).await
    }

    /// Pushes and appends a dataset to the worker.
    ///
    /// # Args
    /// * `xs` - The samples' source.
    /// * `ys` - The labels' source.
    /// * `chunk_size` - The maximum size in bytes for each payload.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_dataset<R>(
        &mut self,
        xs: &mut R,
        ys: &mut R,
        chunk_size: usize,
    ) -> io::Result<()>
    where
        R: AsyncRead + Unpin,
    {
        let mut buf = vec![0; chunk_size];
        self.send_datachunks(xs, &mut buf).await?;
        self.send_datachunks(ys, &mut buf).await?;
        Ok(())
    }

    /// Reads through the reader and sends the datachunks to the worker.
    ///
    /// # Arsg
    /// * `reader` - The byte source.
    /// * `acc` - The accumulating buffer.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn send_datachunks<R>(&mut self, reader: &mut R, acc: &mut [u8]) -> io::Result<()>
    where
        R: AsyncRead + Unpin,
    {
        while utils::read_all(reader, acc).await? > 0 {
            let nums = bytemuck::cast_slice(&acc);
            let msg = Msg::Data(Payload::Datachunk(nums));
            self.transport.send(&msg).await?;
        }

        Ok(())
    }
}
