use std::{env, io};

use comms::{
    msg::{Command, Msg, Payload},
    recv_dataset::{get_dataset_cursor, recv_dataset},
};
use log::{info, warn};
use parameter_server::service::ServerBuilder;
use tokio::{net::TcpListener, signal, task};
use worker::builder::WorkerBuilder;

const DEFAULT_HOST: &str = "0.0.0.0";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(io::Error::other)?;
    let bind_addr = format!("{host}:{port}");

    let listener = TcpListener::bind(&bind_addr).await?;
    info!("listening at {bind_addr}");

    let local = task::LocalSet::new();
    local
        .run_until(async move {
            loop {
                tokio::select! {
                    accept = listener.accept() => {
                        let (stream, orch_addr) = accept?;
                        info!("orchestrator connected from {orch_addr}");

                        let (rx, tx) = stream.into_split();
                        let (mut rx, tx) = comms::channel(rx, tx);

                        let first_msg = match rx.recv().await {
                            Ok(msg) => msg,
                            Err(e) => {
                                warn!("error reading first message from {orch_addr}: {e}");
                                continue;
                            }
                        };

                        match first_msg {
                            Msg::Control(Command::CreateWorker(spec)) => {
                                info!("bootstrapping worker session for {orch_addr}");
                                let bind_addr_clone = bind_addr.clone();
                                task::spawn_local(async move {
                                    let x_size_bytes = spec.dataset.x_size_bytes as usize;
                                    let y_size_bytes = spec.dataset.y_size_bytes as usize;
                                    let mut samples_raw =
                                        vec![0f32; x_size_bytes / size_of::<f32>()];
                                    let mut labels_raw =
                                        vec![0f32; y_size_bytes / size_of::<f32>()];

                                    if let Err(e) = recv_dataset(
                                        &mut get_dataset_cursor(&mut samples_raw),
                                        &mut get_dataset_cursor(&mut labels_raw),
                                        x_size_bytes,
                                        y_size_bytes,
                                        &mut rx,
                                    )
                                    .await
                                    {
                                        warn!("failed to receive dataset: {e}");
                                        return;
                                    }

                                    let worker = match WorkerBuilder::new()
                                        .build(spec, bind_addr_clone, samples_raw, labels_raw)
                                        .await
                                    {
                                        Ok(w) => w,
                                        Err(e) => {
                                            warn!("failed to build worker: {e}");
                                            return;
                                        }
                                    };

                                    if let Err(e) = worker.run(rx, tx).await {
                                        warn!("worker session error: {e}");
                                    } else {
                                        info!("worker session complete");
                                    }
                                });
                            }
                            Msg::Control(Command::CreateServer(spec)) => {
                                info!("bootstrapping parameter server session for {orch_addr}");
                                let nworkers = spec.nworkers;
                                let mut pserver = match ServerBuilder::new()
                                    .build(spec)
                                    .map_err(io::Error::other)
                                {
                                    Ok(p) => p,
                                    Err(e) => {
                                        warn!("failed to build server: {e}");
                                        continue;
                                    }
                                };

                                for i in 0..nworkers {
                                    match listener.accept().await {
                                        Ok((stream, worker_addr)) => {
                                            info!(
                                                "worker {i}/{nworkers} connected from {worker_addr}"
                                            );
                                            let (rx, tx) = stream.into_split();
                                            pserver.spawn(rx, tx);
                                        }
                                        Err(e) => {
                                            warn!(
                                                "failed to accept worker connection {i}: {e}"
                                            );
                                            break;
                                        }
                                    }
                                }

                                tokio::spawn(async move {
                                    let mut tx = tx;
                                    match pserver.run().await {
                                        Ok(mut params) => {
                                            info!(
                                                "server session complete, sending parameters"
                                            );
                                            let msg = Msg::Data(Payload::Params(&mut params));
                                            if let Err(e) = tx.send(&msg).await {
                                                warn!("failed to send parameters: {e}");
                                            }
                                        }
                                        Err(e) => {
                                            warn!("server session error: {e}");
                                        }
                                    }
                                });
                            }
                            msg => {
                                warn!("unexpected first message from {orch_addr}: {msg:?}");
                            }
                        }
                    }
                    _ = signal::ctrl_c() => {
                        info!("received SIGTERM");
                        break;
                    }
                }
            }

            Ok(())
        })
        .await
}
