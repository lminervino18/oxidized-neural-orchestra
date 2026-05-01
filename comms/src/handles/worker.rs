use std::io;

use rand::{SeedableRng, rngs::StdRng};
use tokio::io::AsyncRead;

use super::{Compressor, compressor::CompressedGrad};
use crate::{
    Float01,
    protocol::{Command, Msg, Payload},
    share_dataset, sparse,
    specs::worker::WorkerSpec,
    transport::TransportLayer,
};

/// The handle for communicating with a `Worker`.
pub struct WorkerHandle<T> {
    id: usize,
    transport: T,
    grad: Vec<f32>,
    compressor: Compressor<StdRng>,
}

/// A notified worker event.
#[derive(Debug)]
pub enum WorkerEvent<'a> {
    Grad(&'a [f32]),
    Loss(Vec<f32>),
    RequestParams,
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
            compressor: Compressor::new(),
        }
    }

    /// Enables the sparse gradient capability for this handle.
    ///
    /// # Args
    /// * `r` - The ratio of compression for calculating the threshold value.
    /// * `seed` - The seed for the random number generator.
    pub fn enable_sparse_capability<S>(&mut self, r: Float01, seed: S)
    where
        S: Into<Option<u64>>,
    {
        let rng = match seed.into() {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        self.compressor.enable_sparse_compression(r, rng);
    }

    /// Sends the create spec for the worker.
    ///
    /// # Args
    /// * `spec` - The worker's specification.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn create(&mut self, spec: WorkerSpec) -> io::Result<()> {
        let msg = Msg::Control(Command::CreateWorker(spec));
        self.transport.send(&msg).await
    }

    /// Blocks until receiving an event from a worker.
    ///
    /// # Returns
    /// A `WorkerEvent` message or an io error if occurred.
    pub async fn recv_event(&mut self) -> io::Result<WorkerEvent<'_>> {
        let response = match self.transport.recv().await? {
            Msg::Data(Payload::DenseGrad(grad)) => {
                if let Some(additional) = grad.len().checked_sub(self.grad.capacity()) {
                    self.grad.reserve(additional);
                }

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
            Msg::Control(Command::ReportLoss { losses }) => WorkerEvent::Loss(losses.into_owned()),
            Msg::Control(Command::RequestParams) => WorkerEvent::RequestParams,
            Msg::Control(Command::Disconnect) => WorkerEvent::Disconnect,
            msg => {
                let text = format!("Unexpected message from worker {}, got: {msg:?}", self.id);
                return Err(io::Error::other(text));
            }
        };

        Ok(response)
    }

    /// Pulls the latest parameters from the worker.
    ///
    /// # Returns
    /// The parameters as a mutable slice or an io error if occurred.
    pub async fn pull_params(&mut self) -> io::Result<&mut [f32]> {
        let msg = Msg::Control(Command::RequestParams);
        self.transport.send(&msg).await?;

        let msg = self.transport.recv().await?;
        let Msg::Data(Payload::Params(params)) = msg else {
            let text = format!("Expected params from worker {}, got: {msg:?}", self.id);
            return Err(io::Error::other(text));
        };

        Ok(params)
    }

    /// Pushes the gradient to the worker.
    ///
    /// # Args
    /// * `residual` - The gradient to send.
    ///
    /// # Returns
    /// Either `Some(threshold)` if sparse gradient was used or `None` if dense gradient was used.
    /// Or an io error if occurred.
    pub async fn push_grad(&mut self, residual: &[f32]) -> io::Result<Option<f32>> {
        let (payload, threshold) = match self.compressor.compress(residual) {
            CompressedGrad::Dense { grad } => (Payload::DenseGrad(grad), None),
            CompressedGrad::Sparse { sparse, threshold } => {
                (Payload::SparseGrad(sparse), Some(threshold))
            }
        };

        let msg = Msg::Data(payload);
        self.transport.send(&msg).await?;
        Ok(threshold)
    }

    /// Pushes the latest state of the parameters to the worker.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_params(&mut self, params: &mut [f32]) -> io::Result<()> {
        let msg = Msg::Data(Payload::Params(params));
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
        share_dataset::send_dataset(xs, ys, chunk_size, &mut self.transport).await
    }

    /// Tells the worker to stop it's execution.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn stop(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::StopAfterEpoch);
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
}
