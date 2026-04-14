use std::io;

use half::f16;
use rand::Rng;

use crate::{
    protocol::{Command, Msg, Payload},
    sparse::{self, Float01},
    transport::TransportLayer,
};

/// The handle for communicating with a `ParameterServer`.
pub struct ParamServerHandle<R, L> {
    id: usize,
    transport: L,
    last_threshold: Option<f32>,
    sparse_capability: Option<SparseMetadata<R>>,
    compression_buf: Vec<f16>,
}

/// The response to the parameter pull.
pub enum PullParamsResponse<'a> {
    Params(&'a mut [f32]),
}

/// The necessary metadata to enable the sparse gradient capability.
struct SparseMetadata<R> {
    r: Float01,
    rng: R,
    ser_buf: Vec<u8>,
}

impl<R, L> ParamServerHandle<R, L>
where
    R: Rng,
    L: TransportLayer,
{
    /// Creates a new `ParamServerHandle`.
    ///
    /// # Args
    /// * `id` - The id number of the server.
    /// * `transport` - The transport layer of the communication.
    ///
    /// # Returns
    /// A new `ParamServerHandle` instance.
    pub fn new(id: usize, transport: L) -> Self {
        Self {
            id,
            transport,
            last_threshold: None,
            sparse_capability: None,
            compression_buf: Vec::new(),
        }
    }

    /// Creates a new `ParmaServerHandle` with the sparse gradient capability.
    ///
    /// # Args
    /// * `id` - The id number of the server.
    /// * `transport` - The transport layer of the communication.
    /// * `r` - The ratio of compression for calculating the threshold value.
    /// * `rng` - A random number generator.
    ///
    /// # Returns
    /// A new `ParamServerHandle` instance.
    pub fn with_sparse_capability(id: usize, transport: L, r: Float01, rng: R) -> Self {
        Self {
            id,
            transport,
            last_threshold: None,
            sparse_capability: Some(SparseMetadata {
                r,
                rng,
                ser_buf: Vec::new(),
            }),
            compression_buf: Vec::new(),
        }
    }

    /// Pulls the latest parameters from the server.
    ///
    /// # Returns
    /// The parameters as a mutable slice or an io error if occurred.
    pub async fn pull_params(&mut self) -> io::Result<&mut [f32]> {
        let msg = Msg::Control(Command::RequestParams);
        self.transport.send(&msg).await?;

        let msg = self.transport.recv().await?;
        let Msg::Data(Payload::Params(params)) = msg else {
            let text = format!("Expected params from server {}, got: {msg:?}", self.id);
            return Err(io::Error::other(text));
        };

        Ok(params)
    }

    /// Pushes the gradient to the server.
    ///
    /// # Args
    /// * `residual` - The gradient to send.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_grads(&mut self, residual: &[f32]) -> io::Result<()> {
        let payload = match self.sparse_capability.as_mut() {
            Some(cap) => {
                cap.ser_buf.clear();

                let threshold = sparse::calculate_threshold(residual, cap.r, &mut cap.rng);
                sparse::grad_drop_into(&mut cap.ser_buf, residual, threshold);

                if cap.ser_buf.len() <= residual.len() * size_of::<f16>() {
                    self.last_threshold = Some(threshold);
                    Payload::SparseGrad(&cap.ser_buf)
                } else {
                    self.compress_dense_grad(residual);
                    Payload::Grad(&self.compression_buf)
                }
            }
            None => {
                self.compress_dense_grad(residual);
                Payload::Grad(&self.compression_buf)
            }
        };

        let msg = Msg::Data(payload);
        self.transport.send(&msg).await
    }

    /// Zeroes out the residual buffer using the latest threshold value.
    ///
    /// # Args
    /// * `residual` - The gradient buffer to zero out.
    pub fn zero_residual(&mut self, residual: &mut [f32]) {
        match self.last_threshold.take() {
            None => residual.fill(0.0),
            Some(t) => {
                for g in residual.iter_mut() {
                    if g.abs() >= t {
                        *g = 0.0;
                    }
                }
            }
        }
    }

    /// Disconnects the parameter server.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Disconnect);
        self.transport.send(&msg).await
    }

    /// Compresses the given gradient buffer into the inner `compression_buf`.
    ///
    /// # Args
    /// * `residual` - The gradient to compress.
    fn compress_dense_grad(&mut self, residual: &[f32]) {
        let additional = self
            .compression_buf
            .capacity()
            .saturating_sub(residual.len());

        self.compression_buf.reserve(additional);

        // SAFETY: The new uninitialized bytes will be overwritten right
        //         after with the compressed 16 bit gradient values.
        unsafe { self.compression_buf.set_len(residual.len()) };

        for (g, r) in self.compression_buf.iter_mut().zip(residual) {
            *g = f16::from_f32(*r);
        }
    }
}
