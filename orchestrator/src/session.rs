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
    servers: Vec<(NetRx, NetTx)>,
    workers: Vec<(NetRx, NetTx)>,
    buf: Vec<u32>,
}

impl Session {
    /// Creates a new `Session`.
    ///
    /// # Arguments
    /// * `workers` - A list of network addressses and worker specification tuples.
    /// * `servers` - A list of network addressses and server specification tuples.
    ///
    /// # Returns
    /// A new `Session` instance.
    pub fn new(
        workers: Vec<(SocketAddr, WorkerSpec)>,
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> io::Result<Self> {
        let runtime = Runtime::new()?;
        let server_chans = runtime.block_on(Self::create_servers(servers))?;
        let worker_chans = runtime.block_on(Self::create_workers(workers))?;

        Ok(Self {
            runtime,
            servers: server_chans,
            workers: worker_chans,
            buf: vec![0; 1028],
        })
    }

    /// Waits until the entire training is finished.
    ///
    /// # Returns
    /// The parameters of the model or an io error if failed to do so.
    pub fn wait(mut self) -> io::Result<Vec<Vec<f32>>> {
        self.runtime.block_on(async move {
            for (mut rx, _) in self.workers {
                while !matches!(
                    rx.recv_into(&mut self.buf).await?,
                    Msg::Control(Command::Disconnect)
                ) {}
            }

            let mut all_params = Vec::with_capacity(self.servers.len());

            for (mut rx, _) in self.servers {
                match rx.recv_into(&mut self.buf).await? {
                    Msg::Data(Payload::Params(params)) => {
                        all_params.push(params.to_vec());
                    }
                    msg => {
                        return Err(io::Error::other(format!(
                            "Received an invalid msg kind: {msg:?}"
                        )));
                    }
                };
            }

            Ok(all_params)
        })
    }

    /// Tries to reach the parameter server and create it using the given specification.
    ///
    /// # Arguments
    /// * `servers` - A list of network addressses and server specification tuples.
    ///
    /// # Returns
    /// The communication channels or an io error if failed to do so.
    async fn create_servers(
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        let mut channels = Vec::with_capacity(servers.len());

        for (addr, spec) in servers {
            let (rx, mut tx) = Self::open_channel(addr).await?;
            let msg = Msg::Control(Command::CreateServer(spec));
            tx.send(&msg).await?;
            channels.push((rx, tx));
        }

        Ok(channels)
    }

    /// Tries to reach the workers and create them using the given specification.
    ///
    /// # Arguments
    /// * `workers` - A list of network addressses and worker specification tuples.
    ///
    /// # Returns
    /// The communication channels or an io error if failed to do so.
    async fn create_workers(
        workers: Vec<(SocketAddr, WorkerSpec)>,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        let mut channels = Vec::with_capacity(workers.len());

        for (addr, spec) in workers {
            let (rx, mut tx) = Self::open_channel(addr).await?;
            let msg = Msg::Control(Command::CreateWorker(spec));
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
