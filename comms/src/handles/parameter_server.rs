use std::{
    fmt::{self, Debug, Formatter},
    io,
    marker::PhantomData,
};

use rand::{SeedableRng, rngs::StdRng};
use tokio::io::{AsyncRead, AsyncWrite};
use uuid::Uuid;

use super::{CompressedGrad, Compressor};
use crate::{
    floats::Float01,
    handles::DatasetSrc,
    protocol::{Command, Msg, Payload},
    share_dataset,
    transport::TransportLayer,
};

/// The handle for communicating with a `ParameterServer`.
pub struct ParamServerHandle<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    id: Uuid,
    transport: T,
    compressor: Compressor<StdRng>,
    _phantom: PhantomData<(R, W)>,
}

impl<R, W, T> Debug for ParamServerHandle<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParamServerHandle")
            .field("id", &self.id)
            .finish()
    }
}

impl<R, W, T> ParamServerHandle<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    /// Creates a new `ParamServerHandle`.
    ///
    /// # Args
    /// * `id` - The id of the server.
    /// * `transport` - The transport layer of the communication.
    ///
    /// # Returns
    /// A new `ParamServerHandle` instance.
    pub fn new(id: Uuid, transport: T) -> Self {
        Self {
            id,
            transport,
            compressor: Compressor::new(),
            _phantom: PhantomData,
        }
    }

    /// The server's id.
    ///
    /// # Returns
    /// The unique user id of the server.
    pub fn id(&self) -> Uuid {
        self.id
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

    /// Pulls the latest parameters from the server.
    ///
    /// # Returns
    /// The parameters as a mutable slice or an io error if occurred.
    pub async fn pull_params(&mut self) -> io::Result<&mut [f32]> {
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

    /// Sends a request for the stored params in a server.
    ///
    /// This method is intended to ask for the final state of the
    /// parameters once the training stage ended.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn req_params(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::RequestParams);
        self.transport.send(&msg).await
    }

    /// Waits for a message and discards it.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn discard_one(&mut self) -> io::Result<()> {
        self.transport.recv().await.map(|_| ())
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

impl<R, W, T> DatasetSrc for ParamServerHandle<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    /// Waits to receive the dataset from the parameter server and writes both samples
    /// and labels to the given writers.
    ///
    /// # Args
    /// * `xs` - The sink for samples.
    /// * `ys` - The sink for labels.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn pull_dataset(&mut self, xs: &mut Vec<f32>, ys: &mut Vec<f32>) -> io::Result<()> {
        share_dataset::recv_dataset(xs, ys, &mut self.transport).await
    }
}
