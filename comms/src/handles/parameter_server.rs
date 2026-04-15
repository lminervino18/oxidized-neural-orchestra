use std::io;

use half::f16;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    protocol::{Command, Msg, Payload},
    sparse::{self, Float01},
    transport::TransportLayer,
};

/// The handle for communicating with a `ParameterServer`.
pub struct ParamServerHandle<T> {
    id: usize,
    transport: T,
    last_threshold: Option<f32>,
    sparse_capability: Option<SparseMetadata>,
    compression_buf: Vec<f16>,
}

/// The response to the parameter pull.
pub enum PullParamsResponse<'a> {
    Params(&'a mut [f32]),
}

/// The necessary metadata to enable the sparse gradient capability.
struct SparseMetadata {
    r: Float01,
    rng: StdRng,
    ser_buf: Vec<u8>,
}

impl<T> ParamServerHandle<T>
where
    T: TransportLayer,
{
    /// Creates a new `ParamServerHandle`.
    ///
    /// # Args
    /// * `id` - The id number of the server.
    /// * `transport` - The transport layer of the communication.
    ///
    /// # Returns
    /// A new `ParamServerHandle` instance.
    pub fn new(id: usize, transport: T) -> Self {
        Self {
            id,
            transport,
            last_threshold: None,
            sparse_capability: None,
            compression_buf: Vec::new(),
        }
    }

    /// Enables the sparse gradient capability for this handle.
    ///
    /// # Args
    /// * `r` - The ratio of compression for calculating the threshold value.
    /// * `seed` - The seed for the random number generator.
    pub fn enable_sparse_capabiliy<S>(&mut self, r: Float01, seed: S)
    where
        S: Into<Option<u64>>,
    {
        let rng = match seed.into() {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let sparse_metadata = SparseMetadata {
            r,
            rng,
            ser_buf: Vec::new(),
        };

        self.sparse_capability = Some(sparse_metadata);
    }

    /// Pulls the latest parameters from the server.
    ///
    /// # Returns
    /// The parameters as a mutable slice or an io error if occurred.
    pub async fn pull_params(&mut self) -> io::Result<PullParamsResponse> {
        let msg = Msg::Control(Command::RequestParams);
        self.transport.send(&msg).await?;

        let msg = self.transport.recv().await?;
        let Msg::Data(Payload::Params(params)) = msg else {
            let text = format!("Expected params from server {}, got: {msg:?}", self.id);
            return Err(io::Error::other(text));
        };

        Ok(PullParamsResponse::Params(params))
    }

    /// Pushes the gradient to the server.
    ///
    /// # Args
    /// * `residual` - The gradient to send.
    ///
    /// # Returns
    /// Either `Some(threshold)` if sparse gradient was used or `None` if dense gradient was used.
    /// Or an io error if occurred.
    pub async fn push_grad(&mut self, residual: &[f32]) -> io::Result<Option<f32>> {
        let (payload, threshold) = match self.sparse_capability.as_mut() {
            Some(cap) => {
                cap.ser_buf.clear();

                let threshold = sparse::calculate_threshold(residual, cap.r, &mut cap.rng);
                sparse::grad_drop_into(&mut cap.ser_buf, residual, threshold);

                if cap.ser_buf.len() <= residual.len() * size_of::<f16>() {
                    self.last_threshold = Some(threshold);
                    let payload = Payload::SparseGrad(&cap.ser_buf);
                    (payload, Some(threshold))
                } else {
                    Self::compress_dense_grad(&mut self.compression_buf, residual);
                    let payload = Payload::Grad(&self.compression_buf);
                    (payload, None)
                }
            }
            None => {
                Self::compress_dense_grad(&mut self.compression_buf, residual);
                let payload = Payload::Grad(&self.compression_buf);
                (payload, None)
            }
        };

        let msg = Msg::Data(payload);
        self.transport.send(&msg).await?;
        Ok(threshold)
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
    /// * `compression_buf`: The buffer where to write the compressed residual gradient.
    /// * `residual` - The gradient to compress.
    fn compress_dense_grad(compression_buf: &mut Vec<f16>, residual: &[f32]) {
        let additional = compression_buf.capacity().saturating_sub(residual.len());
        compression_buf.reserve(additional);

        // SAFETY: The new uninitialized bytes will be overwritten right
        //         after with the compressed 16 bit gradient values.
        unsafe { compression_buf.set_len(residual.len()) };

        for (g, r) in compression_buf.iter_mut().zip(residual) {
            *g = f16::from_f32(*r);
        }
    }
}
