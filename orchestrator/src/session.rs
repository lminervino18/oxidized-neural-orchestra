use std::net::SocketAddr;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use tokio::{
    io,
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    runtime::Runtime,
};

type NetRx = OnoReceiver<OwnedReadHalf>;
type NetTx = OnoSender<OwnedWriteHalf>;

/// Represents an ongoing training session that's running in the background.
/// It lets it's owner interact with the entire system from a single instance.
pub struct Session {
    runtime: Runtime,
    server: (NetRx, NetTx),
    workers: Vec<(NetRx, NetTx)>,
}

impl Session {
    /// Creates a new `Session`.
    ///
    /// # Arguments
    /// * `worker_addrs` - A list of the network addresses for the worker nodes.
    /// * `worker_spec` - The specification for the worker nodes' configuration.
    /// * `server_addr` - The network address for the parameter server node.
    /// * `server_spec` - The specification for the server node's configuration.
    ///
    /// # Returns
    /// A new `Session` instance.
    pub fn new(
        worker_addrs: Vec<SocketAddr>,
        worker_spec: WorkerSpec,
        server_addr: SocketAddr,
        server_spec: ServerSpec,
    ) -> io::Result<Self> {
        let runtime = Runtime::new()?;
        let server = runtime.block_on(Self::create_server(server_addr, server_spec))?;
        let workers = runtime.block_on(Self::create_workers(worker_addrs, worker_spec))?;

        Ok(Self {
            runtime,
            workers,
            server,
        })
    }

    /// Waits until the entire training is finished.
    ///
    /// # Returns
    /// The parameters of the model or an io error if failed to do so.
    pub fn wait(self) -> io::Result<Vec<f32>> {
        self.runtime.block_on(async move {
            for (mut rx, _) in self.workers {
                while !matches!(rx.recv().await?, Msg::Control(Command::Disconnect)) {}
            }

            let (mut rx, mut tx) = self.server;
            let msg = Msg::Control(Command::Disconnect);
            tx.send(&msg).await?;

            match rx.recv().await? {
                Msg::Data(Payload::Params(params)) => Ok(params.to_vec()),
                msg => Err(io::Error::other(format!(
                    "Received an invalid msg kind: {msg:?}"
                ))),
            }
        })
    }

    /// Tries to reach the parameter server and create it using the given specification.
    ///
    /// # Arguments
    /// * `server_addr` - The network address for the parameter server node.
    /// * `server_spec` - The specification for the server node's configuration.
    ///
    /// # Returns
    /// The communication channel or an io error if failed to do so.
    async fn create_server(
        server_addr: SocketAddr,
        server_spec: ServerSpec,
    ) -> io::Result<(NetRx, NetTx)> {
        let (rx, mut tx) = Self::open_channel(server_addr).await?;
        let msg = Msg::Control(Command::CreateServer(server_spec));
        tx.send(&msg).await?;
        Ok((rx, tx))
    }

    /// Tries to reach the workers and create them using the given specification.
    ///
    /// # Arguments
    /// * `worker_addrs` - A list of the network addresses for the worker nodes.
    /// * `worker_spec` - The specification for the worker nodes' configuration.
    ///
    /// # Returns
    /// The communication channels or an io error if failed to do so.
    async fn create_workers(
        worker_addrs: Vec<SocketAddr>,
        worker_spec: WorkerSpec,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        let msg = Msg::Control(Command::CreateWorker(worker_spec));
        let mut channels = Vec::with_capacity(worker_addrs.len());

        for addr in worker_addrs {
            let (rx, mut tx) = Self::open_channel(addr).await?;
            tx.send(&msg).await?;
            channels.push((rx, tx));
        }

        Ok(channels)
    }

    /// Creates a communication channel with some entity through it's network address.
    ///
    /// # Arguments
    /// * `addr` - The network address of some node.
    ///
    /// # Returns
    /// A communication channel or an io error if failed to do so.
    async fn open_channel(addr: SocketAddr) -> io::Result<(NetRx, NetTx)> {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        Ok(comms::channel(rx, tx))
    }
}
