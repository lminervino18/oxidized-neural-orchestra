use std::{borrow::Cow, io};

use super::DatasetSrc;
use crate::{
    protocol::{Command, Msg, Payload},
    share_dataset,
    specs::{
        node::{NodeSpec, StatRequest, StatResponse},
        server::ServerSpec,
    },
    transport::TransportLayer,
};

/// The handle for communicating with an `Orchestrator`.
pub struct OrchHandle<T: TransportLayer> {
    transport: T,
}

/// A notified orchestrator event.
#[derive(Debug)]
pub enum OrchEvent {
    Create {
        spec: NodeSpec,
    },
    Disconnect,
    RequestParams,
    ShareDataset,
    StatsRequest {
        reqs: Vec<StatRequest>,
    },
    Stop,
    Switch {
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    },
    Upgrade {
        spec: ServerSpec,
        ranges: Vec<(usize, usize)>,
    },
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
    pub async fn push_losses(&mut self, losses: &[f64]) -> io::Result<()> {
        if losses.iter().any(|l| !l.is_finite()) {
            return Err(io::Error::other("loss diverged: NaN or Inf detected"));
        }

        let msg = Msg::Control(Command::ReportLoss {
            losses: Cow::Borrowed(losses),
        });

        self.transport.send(&msg).await
    }

    /// Pushes the given statistics onto the orchestrator.
    ///
    /// # Args
    /// * `stats` - The statistic responses.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_stats(&mut self, stats: Vec<StatResponse>) -> io::Result<()> {
        let msg = Msg::Control(Command::StatsResponse { stats });
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
            Msg::Control(Command::CreateNode { spec }) => OrchEvent::Create { spec },
            Msg::Control(Command::Upgrade { spec, ranges }) => OrchEvent::Upgrade { spec, ranges },
            Msg::Control(Command::StatsRequest { reqs }) => OrchEvent::StatsRequest { reqs },
            Msg::Control(Command::Switch {
                server_addrs,
                server_sizes,
                server_ordering,
            }) => OrchEvent::Switch {
                server_addrs,
                server_sizes,
                server_ordering,
            },
            Msg::Control(Command::ShareDataset) => OrchEvent::ShareDataset,
            msg => {
                let text = format!("Unexpected message from orchestrator, got: {msg:?}");
                return Err(io::Error::other(text));
            }
        };

        Ok(event)
    }

    /// Tells the orchestrator that an entity has ended it's processing.
    /// This message is used exclusively by workers.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn done(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Done);
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

impl<T> DatasetSrc for OrchHandle<T>
where
    T: TransportLayer,
{
    /// Waits to receive the dataset from the orchestrator and writes both samples
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
