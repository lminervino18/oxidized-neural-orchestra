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
        self.servers.push(ServerMetadata::new(rx, tx, size));
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
                    Msg::Data(Payload::Params(params)) if params.len() != server.acc_grad_buf.len() => {
                        let (expected, got) = (server.acc_grad_buf.len(), params.len());
                        let text = format!("the length of the received params from server {i} is invalid, expected {expected}, got {got}");
                        Err(io::Error::other(text))
                    }
                    Msg::Data(Payload::Params(params)) => {
                        // SAFETY: The length of both buffers is the same.
                        server.acc_grad_buf.copy_from_slice(params);

                        let metadata = ServerParamsMetadata {
                            params,
                            grad: &mut server.grad,
                            acc_grad_buf: &mut server.acc_grad_buf,
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
            let msg = Msg::Data(Payload::Grad(&server.acc_grad_buf));
            server.tx.send(&msg).await?;

            // TODO: Maybe do this somewhere else.
            server.acc_grad_buf.fill(0.0);
            Ok::<_, io::Error>(())
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
