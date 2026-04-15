use std::{io, time::Duration};

use tokio::io::{AsyncRead, AsyncWrite};

use crate::{
    handles::{OrchHandle, ParamServerHandle, WorkerHandle},
    protocol::{Command, Entity, Msg},
    transport::{self, Rtp, TransportLayer},
};

/// Establishes connections and yields reliable transports.
pub struct Connector {
    timeout: Duration,
    base_retry_dur: Duration,
    retry_coef: u32,
    retries: usize,
    src_entity: Entity,
}

impl Connector {
    /// Creates a new `Connector`.
    ///
    /// # Args
    /// * `timeout` - The timeout duration to wait for the receival of a message.
    /// * `base_retry_dur` - The base duration for the exponential backoff retryer.
    /// * `retry_coef` - The coefficient to which to multiply the current wait duration.
    /// * `retries` - The amount of retries till declaring a dead node.
    /// * `entity` - The callee's entity.
    ///
    /// # Returns
    /// A new `Connector` instance.
    pub fn new(
        timeout: Duration,
        base_retry_dur: Duration,
        retry_coef: u32,
        retries: usize,
        src_entity: Entity,
    ) -> Self {
        Self {
            timeout,
            base_retry_dur,
            retry_coef,
            retries,
            src_entity,
        }
    }

    /// Connects the given channel to a worker using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `id` - The id of the server.
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    ///
    /// # Returns
    /// A new `WorkerHandle` instance.
    pub async fn connect_worker<R, W>(
        &self,
        id: usize,
        reader: R,
        writer: W,
    ) -> io::Result<WorkerHandle<Rtp<R, W>>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer).await?;
        Ok(WorkerHandle::new(id, transport_layer))
    }

    /// Connects the given channel to a parameter server using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `id` - The id of the worker.
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    ///
    /// # Returns
    /// A new `ParamServerHandle` instance.
    pub async fn connect_parameter_server<R, W>(
        &self,
        id: usize,
        reader: R,
        writer: W,
    ) -> io::Result<ParamServerHandle<Rtp<R, W>>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer).await?;
        Ok(ParamServerHandle::new(id, transport_layer))
    }

    /// Connects the given channel to an orchestrator using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    ///
    /// # Returns
    /// A new `OrchHandle` instance.
    pub async fn connect_orchestrator<R, W>(
        &self,
        reader: R,
        writer: W,
    ) -> io::Result<OrchHandle<Rtp<R, W>>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer).await?;
        Ok(OrchHandle::new(transport_layer))
    }

    /// Connects the given channel to an entity using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    ///
    /// # Returns
    /// A new `ReliableTransport` or an io error if occurred.
    async fn connect<R, W>(&self, reader: R, writer: W) -> io::Result<Rtp<R, W>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut transport_layer = transport::build_reliable_transport(
            reader,
            writer,
            self.timeout,
            self.base_retry_dur,
            self.retry_coef,
            self.retries,
        );

        let msg = Msg::Control(Command::Connect(self.src_entity));
        transport_layer.send(&msg).await?;
        Ok(transport_layer)
    }
}
