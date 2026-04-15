use std::{io, time::Duration};

use tokio::io::{AsyncRead, AsyncWrite};

use super::Connection;
use crate::{
    handles::{OrchHandle, ParamServerHandle, WorkerHandle},
    protocol::{Command, Entity, Msg},
    transport::{self, TransportLayer},
};

/// Accepts new incoming connections and assigns yields their handle types.
pub struct Acceptor<R, W, F>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    F: AsyncFnMut() -> io::Result<(R, W)>,
{
    listener: F,
    timeout: Duration,
    base_retry_dur: Duration,
    retry_coef: u32,
    retries: usize,
}

impl<R, W, F> Acceptor<R, W, F>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    F: AsyncFnMut() -> io::Result<(R, W)>,
{
    /// Creates a new `Acceptor`.
    ///
    /// # Args
    /// * `listener` - A stream socket factory factory.
    /// * `timeout` - The timeout duration to declare that a message didn't reach the other end.
    /// * `base_retry_dur` - The duration that the acceptor sleeps for the first time a message timed out.
    /// * `retry_coef` - The coeficient to multiply the current timeout duration.
    /// * `retries` - The amount of retries to make.
    ///
    /// # Returns
    /// A new `Acceptor` instance.
    pub fn new(
        listener: F,
        timeout: Duration,
        base_retry_dur: Duration,
        retry_coef: u32,
        retries: usize,
    ) -> Self {
        Self {
            listener,
            timeout,
            base_retry_dur,
            retry_coef,
            retries,
        }
    }

    /// Blocks the current thread until a new connection arrives.
    ///
    /// # Returns
    /// A new connection or an io error if occurred while waiting for incoming connections
    /// or receiving the type of entity from the peer.
    pub async fn accept(&mut self) -> io::Result<Connection<impl TransportLayer>> {
        let (reader, writer) = (self.listener)().await?;

        let mut transport_layer = transport::build_reliable_transport(
            reader,
            writer,
            self.timeout,
            self.base_retry_dur,
            self.retry_coef,
            self.retries,
        );

        let msg = transport_layer.recv().await?;
        let Msg::Control(Command::Connect(entity)) = msg else {
            let text = format!("Expected Connect message, got: {msg:?}");
            return Err(io::Error::other(text));
        };

        let conn = match entity {
            Entity::Worker { id } => Connection::Worker(WorkerHandle::new(id, transport_layer)),
            Entity::ParamServer { id } => {
                Connection::ParamServer(ParamServerHandle::new(id, transport_layer))
            }
            Entity::Orchestrator => Connection::Orchestrator(OrchHandle::new(transport_layer)),
        };

        Ok(conn)
    }
}
