mod metadata;

use std::io;

use comms::{
    Deserialize, OnoReceiver, OnoSender,
    msg::{Msg, Payload},
};
use futures::future;
use log::warn;
use machine_learning::middleware::{ParamManager, ServerParamsMetadata};
use tokio::io::{AsyncRead, AsyncWrite};

use metadata::ServerMetadata;

const STARTING_RX_BUF_SIZE: usize = 1028;

pub struct ServerManager<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    servers: Vec<ServerMetadata<R, W>>,
    server_ordering: Vec<usize>,
    layer_sizes: Vec<usize>,
}

impl<R, W> ServerManager<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub fn new(server_ordering: Vec<usize>, layer_sizes: Vec<usize>) -> Self {
        Self {
            servers: Vec::new(),
            server_ordering,
            layer_sizes,
        }
    }

    pub fn spawn(&mut self, rx: OnoReceiver<R>, tx: OnoSender<W>, size: usize) {
        let metadata = ServerMetadata {
            rx,
            tx,
            rx_buf: vec![0; STARTING_RX_BUF_SIZE],
            grad: vec![0.0; size],
        };

        self.servers.push(metadata);
    }

    pub async fn param_manager(&mut self) -> io::Result<ParamManager<'_>> {
        let mut servers = Vec::with_capacity(self.servers.len());

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

        Ok(ParamManager::new(
            servers,
            &self.server_ordering,
            &self.layer_sizes,
        ))
    }
}
