use std::{env, io, time::Duration};

use comms::Acceptor;
use log::info;
use tokio::net::TcpListener;

mod router;

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

    let stream_factory = async move || {
        let (stream, peer_addr) = listener.accept().await?;
        info!("new incoming connection from {peer_addr}");
        Ok(stream.into_split())
    };

    let acceptor = Acceptor::new(
        stream_factory,
        Duration::from_secs(5),
        Duration::from_secs(2),
        2,
        5,
    );

    NodeRouter::new(acceptor).run().await
}
