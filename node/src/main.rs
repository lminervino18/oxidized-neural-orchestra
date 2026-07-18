mod router;
mod stat_service;

use std::{env, io};

use comms::{Acceptor, Connector};
use log::info;
use tokio::net::TcpListener;
use tracing_subscriber::{EnvFilter, fmt};

use router::NodeRouter;
use uuid::Uuid;

/// A default host address for the tcp listener in the acceptor.
const DEFAULT_HOST: &str = "0.0.0.0";

/// Initializes stderr logging, defaulting to `info` when `RUST_LOG` is unset.
fn init_logging() {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
}

#[tokio::main]
async fn main() -> io::Result<()> {
    init_logging();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(io::Error::other)?;
    let addr = format!("{host}:{port}");

    let listener = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let stp_factory = comms::build_simple_transport;

    let transport_factory = async || {
        let (stream, peer_addr) = listener.accept().await?;
        info!("new incoming connection from {peer_addr}");

        let (rx, tx) = stream.into_split();
        Ok(stp_factory(rx, tx))
    };

    let id = Uuid::new_v4();
    info!("Assigned node id {id}");

    let acceptor = Acceptor::new(id, transport_factory);
    let connector = Connector::new(id, stp_factory);
    NodeRouter::new(acceptor, connector).run().await
}
