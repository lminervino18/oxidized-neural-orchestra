mod router;

use std::{env, io, time::Duration};

use comms::{Acceptor, Connector};
use log::info;
use tokio::net::TcpListener;

use router::NodeRouter;

const DEFAULT_HOST: &str = "0.0.0.0";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(io::Error::other)?;
    let addr = format!("{host}:{port}");

    let listener = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let transport_factory = async || {
        let (stream, peer_addr) = listener.accept().await?;
        info!("new incoming connection from {peer_addr}");

        let (rx, tx) = stream.into_split();
        let transport_layer = comms::build_reliable_transport(
            rx,
            tx,
            Duration::from_secs(5),
            Duration::from_secs(2),
            2,
            5,
        );

        Ok(transport_layer)
    };
    let acceptor = Acceptor::new(transport_factory);

    let transport_factory = |rx, tx| {
        comms::build_reliable_transport(
            rx,
            tx,
            Duration::from_secs(5),
            Duration::from_secs(2),
            2,
            5,
        )
    };
    let connector = Connector::new(transport_factory);

    NodeRouter::new(acceptor, connector).run().await
}
