mod router;

use std::{env, io, time::Duration};

use comms::{Acceptor, Connector};
use log::info;
use tokio::net::TcpListener;

use router::NodeRouter;

/// A default host address for the tcp listener in the acceptor.
const DEFAULT_HOST: &str = "0.0.0.0";

/// The timeout duration for the reliable transport.
const NETWORK_TIMEOUT: Duration = Duration::from_secs(5);

/// The starting sleep duration for exponential backoff.
const NETWORK_EXP_BACKOFF_BASE: Duration = Duration::from_secs(2);

/// The coeficient which to multiply the current sleep duration for exponential backoff.
const NETWORK_EXP_BACKOFF_COEF: u32 = 2;

/// The amount of retries to do until giving up the connection for exponential backoff.
const NETWORK_EXP_BACKOFF_RETRIES: usize = 4;

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(io::Error::other)?;
    let addr = format!("{host}:{port}");

    let listener = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let rtp_factory = |rx, tx| {
        comms::build_reliable_transport(
            rx,
            tx,
            NETWORK_TIMEOUT,
            NETWORK_EXP_BACKOFF_BASE,
            NETWORK_EXP_BACKOFF_COEF,
            NETWORK_EXP_BACKOFF_RETRIES,
        )
    };

    let transport_factory = async || {
        let (stream, peer_addr) = listener.accept().await?;
        info!("new incoming connection from {peer_addr}");

        let (rx, tx) = stream.into_split();
        Ok(rtp_factory(rx, tx))
    };

    let acceptor = Acceptor::new(transport_factory);
    let connector = Connector::new(rtp_factory);
    NodeRouter::new(acceptor, connector).run().await
}
