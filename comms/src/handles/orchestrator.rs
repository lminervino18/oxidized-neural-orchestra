use std::{borrow::Cow, io};

use tokio::io::AsyncWrite;

use crate::{
    protocol::{Command, Msg, Payload},
    share_dataset,
    specs::{server::ServerSpec, worker::WorkerSpec},
    transport::TransportLayer,
};

/// The handle for communicating with an `Orchestrator`.
pub struct OrchHandle<T: TransportLayer> {
    transport: T,
}

/// The response of pulling a node specification.
pub enum PullSpecResponse {
    Worker(WorkerSpec),
    ParameterServer(ServerSpec),
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

    pub async fn pull_specification(&mut self) -> io::Result<PullSpecResponse> {
        let spec = match self.transport.recv().await? {
            Msg::Control(Command::CreateServer(spec)) => PullSpecResponse::ParameterServer(spec),
            Msg::Control(Command::CreateWorker(spec)) => PullSpecResponse::Worker(spec),
            msg => {
                let text = format!("Expected creation from orchestrator, got: {msg:?}");
                return Err(io::Error::other(text));
            }
        };

        Ok(spec)
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

    /// Pushes the latest parameters to the orchestrator.
    ///
    /// # Args
    /// * `params` - The parameters to push.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_params(&mut self, params: &mut [f32]) -> io::Result<()> {
        let msg = Msg::Data(Payload::Params(params));
        self.transport.send(&msg).await
    }

    /// Pushes the latest losses to the orchestrator.
    ///
    /// # Args
    /// * `losses` - An array of loss values.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_losses(&mut self, losses: &[f32]) -> io::Result<()> {
        let msg = Msg::Control(Command::ReportLoss {
            losses: Cow::Borrowed(losses),
        });

        self.transport.send(&msg).await
    }

    /// Disconnects the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Disconnect);
        self.transport.send(&msg).await
    }
}
