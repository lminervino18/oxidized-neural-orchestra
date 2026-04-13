use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
};
use futures::future;
use tokio::io::{AsyncRead, AsyncWrite};

/// The necessary information to maintain for the entire
/// training duration for each of the servers.
struct ServerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    rx: OnoReceiver<R>,
    tx: OnoSender<W>,
    grad: Vec<f32>,
    residual: Vec<f32>,
}

/// The pulled parameter buffers for a single parameter-server shard.
pub struct PulledServerParams<'a> {
    pub params: &'a mut [f32],
    pub grad: &'a mut [f32],
    pub residual: &'a mut [f32],
}

/// The communication manager between the worker process and the parameter servers.
pub struct ParameterServerMiddleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    servers: Vec<ServerMetadata<R, W>>,
    server_ordering: Vec<usize>,
}

impl<R, W> ParameterServerMiddleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `ParameterServerMiddleware`.
    ///
    /// # Args
    /// * `server_ordering` - The ordering of the servers to know which layer's parameters
    ///   correspond to which server.
    ///
    /// # Returns
    /// A new `ParameterServerMiddleware` instance.
    pub fn new(server_ordering: Vec<usize>) -> Self {
        Self {
            servers: Vec::new(),
            server_ordering,
        }
    }

    /// Adds a new server communicator to the middleware.
    ///
    /// # Args
    /// * `rx` - The worker's receiving end of the communication.
    /// * `tx` - The worker's sending end of the communication.
    /// * `size` - The amount of parameters this server holds.
    pub fn spawn(&mut self, rx: OnoReceiver<R>, tx: OnoSender<W>, size: usize) {
        let metadata = ServerMetadata {
            rx,
            tx,
            grad: vec![0.0; size],
            residual: vec![0.0; size],
        };

        self.servers.push(metadata);
    }

    /// Pulls the new parameters from all the servers.
    ///
    /// # Returns
    /// The parameter, gradient and residual buffers for all server shards.
    ///
    /// # Errors
    /// Returns an io error if a server sends an unexpected message.
    pub async fn pull_params(&mut self) -> io::Result<Vec<PulledServerParams<'_>>> {
        let futs = self
            .servers
            .iter_mut()
            .enumerate()
            .map(async |(i, server)| match server.rx.recv().await? {
                Msg::Data(Payload::Params(params)) => Ok(PulledServerParams {
                    params,
                    grad: &mut server.grad,
                    residual: &mut server.residual,
                }),
                msg => {
                    let text = format!("expected params from server {i}, got: {msg:?}");
                    Err(io::Error::other(text))
                }
            });

        future::try_join_all(futs).await
    }

    /// Returns the ordering of the servers to know which layer's parameters
    /// correspond to which server.
    pub fn server_ordering(&self) -> &[usize] {
        &self.server_ordering
    }

    /// Pushes the latest accumulated gradients to all servers.
    ///
    /// # Errors
    /// Returns an io error if any server cannot receive the gradients.
    pub async fn push_grads(&mut self) -> io::Result<()> {
        let futs = self.servers.iter_mut().map(async |server| {
            let msg = Msg::Data(Payload::Grad(&server.residual));
            let threshold = server.tx.send(&msg).await?;
            Ok::<_, io::Error>(threshold)
        });

        let thresholds = future::try_join_all(futs).await?;

        for (server, threshold) in self.servers.iter_mut().zip(thresholds) {
            let residual = &mut server.residual;

            match threshold {
                None => residual.fill(0.0),
                Some(t) => {
                    residual
                        .iter_mut()
                        .filter(|g| g.abs() >= t)
                        .for_each(|g| *g = 0.0);
                }
            }
        }

        Ok(())
    }

    /// Disconnects this worker from all parameter servers.
    ///
    /// # Errors
    /// Returns an io error if any disconnect exchange fails.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Disconnect);

        let futs = self.servers.iter_mut().map(async |server| {
            server.tx.send(&msg).await?;

            while !matches!(server.rx.recv().await?, Msg::Control(Command::Disconnect)) {}

            Ok::<_, io::Error>(())
        });

        future::try_join_all(futs).await?;
        Ok(())
    }
}
