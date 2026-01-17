mod initialization;
mod optimization;
mod service;
mod storage;
mod training;

use std::{env, error::Error};

use comms::msg::{Command, Msg, Payload};
use tokio::{net::TcpListener, signal};

use crate::service::ServerBuilder;

const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8765";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT")
        .unwrap_or_else(|_| DEFAULT_PORT.to_string())
        .parse::<u16>()?;

    let addr = format!("{host}:{port}");
    let list = TcpListener::bind(addr).await?;
    let builder = ServerBuilder::new();

    let (stream, _) = list.accept().await?;
    let (rx, tx) = stream.into_split();
    let (mut rx, mut tx) = comms::channel(rx, tx);

    let spec = loop {
        let msg = match rx.recv().await {
            Ok(msg) => msg,
            Err(e) => {
                println!("{e:?}");
                continue;
            }
        };

        let Msg::Control(Command::CreateServer(spec)) = msg else {
            println!("Expected CreateServer");
            continue;
        };

        break spec;
    };

    let workers = spec.workers;
    let mut pserver = builder.build(spec)?;

    for _ in 0..workers {
        let (stream, _) = list.accept().await?;
        let (rx, tx) = stream.into_split();
        let (rx, tx) = comms::channel(rx, tx);
        pserver.spawn(rx, tx);
    }

    tokio::select! {
        ret = pserver.run() => {
            let mut weights = ret?;
            let msg = Msg::Data(Payload::Weights(&mut weights));
            tx.send(&msg).await?;
        },
        _ = signal::ctrl_c() => {},
    }

    Ok(())
}
