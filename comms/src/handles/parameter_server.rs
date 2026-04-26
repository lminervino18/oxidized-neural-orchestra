use std::io;

use rand::{SeedableRng, rngs::StdRng};

use super::{CompressedGrad, Compressor};
use crate::{
    protocol::{Command, Msg, Payload},
    sparse::Float01,
    specs::server::ServerSpec,
    transport::TransportLayer,
};

/// The handle for communicating with a `ParameterServer`.
pub struct ParamServerHandle<T> {
    id: usize,
    transport: T,
    compressor: Compressor<StdRng>,
}

/// The response to the parameter pull.
pub enum PullParamsResponse<'a> {
    Params(&'a mut [f32]),
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

    /// Sends the create spec for the server.
    ///
    /// # Args
    /// * `spec` - The server's specification.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn create(&mut self, spec: ServerSpec) -> io::Result<()> {
        let msg = Msg::Control(Command::CreateServer(spec));
        self.transport.send(&msg).await
    }

    /// Pulls the latest parameters from the server.
    ///
    /// # Returns
    /// The parameters as a mutable slice or an io error if occurred.
    pub async fn pull_params(&mut self) -> io::Result<PullParamsResponse<'_>> {
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

    /// Disconnects the parameter server.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Disconnect);
        self.transport.send(&msg).await
    }
}
