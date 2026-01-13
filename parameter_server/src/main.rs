mod initialization;
mod optimization;
mod service;
mod storage;
mod training;

use std::error::Error;

use comms::msg::{Command, Msg, Payload};
use tokio::{net::TcpListener, signal};

use crate::service::ServerBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    const ADDR: &str = "127.0.0.1:8765";

    let list = TcpListener::bind(ADDR).await?;
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
        Ok(mut weights) = pserver.train() => {
            let msg = Msg::Data(Payload::Weights(&mut weights));
            tx.send(&msg).await?;
        },
        _ = signal::ctrl_c() => {},
    }

    Ok(())
}
