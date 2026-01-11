mod initialization;
mod optimization;
mod service;
mod storage;
mod training;

use std::io;

use comms::msg::{Command, Msg};
use tokio::{net::TcpListener, signal};

use crate::service::ServerBuilder;

#[tokio::main]
async fn main() -> io::Result<()> {
    const ADDR: &str = "127.0.0.1:8765";

    let list = TcpListener::bind(ADDR).await?;
    let builder = ServerBuilder::new();

    let (stream, _) = list.accept().await?;
    let (rx, tx) = stream.into_split();
    let (mut rx, _) = comms::channel(rx, tx);

    loop {
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

        let workers = spec.workers;
        let mut pserver = match builder.build(spec) {
            Ok(pserver) => pserver,
            Err(e) => {
                println!("{e:?}");
                continue;
            }
        };

        for _ in 0..workers {
            let (stream, _) = list.accept().await?;
            let (rx, tx) = stream.into_split();
            let (rx, tx) = comms::channel(rx, tx);
            pserver.spawn(rx, tx);
        }

        tokio::select! {
            _ = pserver.run() => break,
            _ = signal::ctrl_c() => {}
        }
    }

    Ok(())
}
