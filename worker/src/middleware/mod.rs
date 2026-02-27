mod metadata;

use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
};
use futures::future;
use machine_learning::middleware::{ParamManager, ServerParamsMetadata};
use tokio::io::{AsyncRead, AsyncWrite};

use metadata::ServerMetadata;

// The starting size of the receiver buffer.
const STARTING_RX_BUF_SIZE: usize = 1028;

// The communication manager between the worker process and the many servers.
pub struct Middleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    servers: Vec<ServerMetadata<R, W>>,
    server_ordering: Vec<usize>,
}

impl<R, W> Middleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `Middleware`.
    ///
    /// # Arguments
    /// * `server_ordering`: The ordering of the servers to know which layer's parameters corresponds to which server.
    ///
    /// # Returns
    /// A new `Middleware` instance.
    pub fn new(server_ordering: Vec<usize>) -> Self {
        Self {
            servers: Vec::new(),
            server_ordering,
        }
    }

    /// Adds a new server communicator to the middleware.
    ///
    /// # Arguments
    /// * `rx` - The worker's receiving end of the communication.
    /// * `tx` - The worker's sending end of the communication.
    /// * `size` - The amount of parameters this server holds.
    pub fn spawn(&mut self, rx: OnoReceiver<R>, tx: OnoSender<W>, size: usize) {
        let metadata = ServerMetadata {
            rx,
            tx,
            rx_buf: vec![0; STARTING_RX_BUF_SIZE],
            grad: vec![0.0; size],
        };

        self.servers.push(metadata);
    }

    /// Pulls the new parameters from all the servers.
    ///
    /// # Returns
    /// A new `ParamManager` instance with all the parameters.
    pub async fn pull_params(&mut self) -> io::Result<ParamManager<'_>> {
        let futs = self
            .servers
            .iter_mut()
            .enumerate()
            .map(
                async |(i, server)| match server.rx.recv_into(&mut server.rx_buf).await? {
                    Msg::Data(Payload::Params(params)) => {
                        let metadata = ServerParamsMetadata {
                            params,
                            grad: &mut server.grad,
                        };

                        Ok(metadata)
                    }
                    msg => {
                        let text = format!("expected params from server {i}, got: {msg:?}");
                        Err(io::Error::other(text))
                    }
                },
            );

        let servers = future::try_join_all(futs).await?;
        Ok(ParamManager::new(servers, &self.server_ordering))
    }

    /// Pushes the latest gradients to the servers.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn push_grads(&mut self) -> io::Result<()> {
        let futs = self.servers.iter_mut().map(async |server| {
            let msg = Msg::Data(Payload::Grad(&server.grad));
            server.tx.send(&msg).await
        });

        future::try_join_all(futs).await?;
        Ok(())
    }

    /// Disconnects this worker from all the servers.
    ///
    /// # Returns
    /// an io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Disconnect);

        let futs = self.servers.iter_mut().map(async |server| {
            server.tx.send(&msg).await?;

            while !matches!(
                server.rx.recv_into(&mut server.rx_buf).await?,
                Msg::Control(Command::Disconnect)
            ) {}

            Ok::<_, io::Error>(())
        });

        future::try_join_all(futs).await?;
        Ok(())
    }
}
