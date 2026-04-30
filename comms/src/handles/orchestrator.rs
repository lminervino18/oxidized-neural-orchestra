use std::{borrow::Cow, io};

use tokio::io::AsyncWrite;

use crate::{
    protocol::{Command, Msg, Payload},
    share_dataset,
    specs::{node::NodeSpec, server::ServerSpec, worker::WorkerSpec},
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

/// A notified orchestrator event.
#[derive(Debug, Clone, Copy)]
pub enum OrchEvent {
    Disconnect,
    RequestParams,
    Stop,
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

    /// Waits till the orchestrator sends the specification for this node.
    ///
    /// # Returns
    /// The specification or an io error if occurred.
    pub async fn pull_specification(&mut self) -> io::Result<PullSpecResponse> {
        let spec = match self.transport.recv().await? {
            Msg::Control(Command::CreateNode(NodeSpec::Server(spec))) => {
                PullSpecResponse::ParameterServer(spec)
            }
            Msg::Control(Command::CreateNode(NodeSpec::Worker(spec))) => {
                PullSpecResponse::Worker(spec)
            }
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
        if losses.iter().any(|l| !l.is_finite()) {
            return Err(io::Error::other("loss diverged: NaN or Inf detected"));
        }

        let msg = Msg::Control(Command::ReportLoss {
            losses: Cow::Borrowed(losses),
        });

        self.transport.send(&msg).await
    }

    /// Blocks until receiving an event from an orchestrator.
    ///
    /// # Returns
    /// An `OrchEvent` message or an io error if occurred.
    pub async fn recv_event(&mut self) -> io::Result<OrchEvent> {
        let event = match self.transport.recv().await? {
            Msg::Control(Command::Disconnect) => OrchEvent::Disconnect,
            Msg::Control(Command::RequestParams) => OrchEvent::RequestParams,
            Msg::Control(Command::StopAfterEpoch) => OrchEvent::Stop,
            msg => {
                let text = format!("Unexpected message from orchestrator, got: {msg:?}");
                return Err(io::Error::other(text));
            }
        };

        Ok(event)
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
