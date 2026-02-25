mod metadata;

use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Msg, Payload},
};
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
    layer_sizes: Vec<usize>,
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
    /// * `layer_sizes`: The amount of parameters per layer of the model.
    ///
    /// # Returns
    /// A new `Middleware` instance.
    pub fn new(server_ordering: Vec<usize>, layer_sizes: Vec<usize>) -> Self {
        Self {
            servers: Vec::new(),
            server_ordering,
            layer_sizes,
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
        let mut servers = Vec::with_capacity(self.servers.len());

        // TODO: paralelizar esto
        for (i, server) in self.servers.iter_mut().enumerate() {
            match server.rx.recv_into(&mut server.rx_buf).await? {
                Msg::Data(Payload::Params(params)) => {
                    let metadata = ServerParamsMetadata {
                        params,
                        grad: &mut server.grad,
                    };

                    servers.push(metadata);
                }
                msg => {
                    let text = format!("expected Params from server {i}, got: {msg:?}");
                    return Err(io::Error::other(text));
                }
            }
        }

        Ok(ParamManager::new(servers, &self.server_ordering))
    }

    pub async fn push_grads(&mut self) -> io::Result<()> {
        // TODO: Paralelizar esto
        for server in self.servers.iter_mut() {
            let msg = Msg::Data(Payload::Grad(&server.grad));
            server.tx.send(&msg).await?;
        }

        Ok(())
    }
}
