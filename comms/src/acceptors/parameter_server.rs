use std::{io, time::Duration};

use rand::Rng;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::TcpListener,
};

use crate::{
    handles::{OrchHandle, ParamServerHandle, WorkerHandle},
    protocol::{Command, Msg, SrcEntity},
    transport::{Framer, Retryer, TimeOuter, TransportLayer},
};

pub struct Acceptor {
    listener: TcpListener,
    timeout: Duration,
    base_retry_dur: Duration,
    retry_coef: u32,
    retries: usize,
}

pub enum Connection<T, R>
where
    T: TransportLayer,
    R: Rng,
{
    Worker(WorkerHandle<T>),
    ParamServer(ParamServerHandle<R, T>),
    Orchestrator(OrchHandle<T>),
}

impl Acceptor {
    pub fn new(
        listener: TcpListener,
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
    pub async fn accept<T, R>(&mut self) -> io::Result<Connection<T, R>>
    where
        T: TransportLayer,
        R: Rng,
    {
        let (stream, addr) = self.listener.accept().await?;
        let (reader, writer) = stream.into_split();
        let mut transport = self.build_reliable_transport(reader, writer);

        let msg = transport.recv().await?;
        let Msg::Control(Command::Connect(entity)) = msg else {
            return Err(io::Error::other("Expected Connect message, got: {msg:?}"));
        };

        let conn = match entity {
            SrcEntity::Worker { id } => {
                let worker_handle = WorkerHandle::new(id, transport);
                Connection::Worker(worker_handle)
            }
            SrcEntity::ParamServer {
                id,
                sparse_cabaple: false,
            } => {
                let param_server_handle = ParamServerHandle::new(id, transport);
                Connection::ParamServer(param_server_handle)
            }
            SrcEntity::Orchestrator => {
                let orch_handle = OrchHandle::new(transport);
                Connection::Orchestrator(orch_handle)
            }
            _ => todo!(),
        };

        Ok(conn)
    }

    /// Builds a reliable connection with timeout and retry capabilities.
    ///
    /// # Args
    /// * `reader` - The inner reader for the connection.
    /// * `writer` - The inner writer for the connection.
    ///
    /// # Returns
    /// The reliable transport layer.
    fn build_reliable_transport<R, W>(&self, reader: R, writer: W) -> impl TransportLayer
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let framer = Framer::new(reader, writer);
        let timeouter = TimeOuter::new(self.timeout, framer);
        let retryer = Retryer::new(
            self.base_retry_dur,
            self.retry_coef,
            self.retries,
            timeouter,
        );

        retryer
    }
}
