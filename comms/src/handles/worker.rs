use std::io;

use crate::{
    protocol::{Command, Msg, Payload},
    sparse,
    transport::TransportLayer,
};

/// The handle for communicating with a `Worker`.
pub struct WorkerHandle<T> {
    id: usize,
    transport: T,
    grad: Vec<f32>,
}

/// The response to the gradient pull.
pub enum PullGradResponse<'a> {
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

    /// Blocks until receiving a message from a worker, it expects a gradient.
    ///
    /// # Returns
    /// A `PullGradResponse` message or an io error if occurred.
    pub async fn pull_grad(&mut self) -> io::Result<PullGradResponse> {
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

                PullGradResponse::Grad(&self.grad)
            }
            Msg::Data(Payload::SparseGrad(grad)) => {
                sparse::grad_lift_into(&mut self.grad, grad).map_err(io::Error::other)?;
                PullGradResponse::Grad(&self.grad)
            }
            Msg::Control(Command::Disconnect) => PullGradResponse::Disconnect,
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
}
