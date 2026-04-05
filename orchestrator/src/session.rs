use futures::future;
use log::{debug, error, info, warn};
use std::{
    io::{self, Cursor},
    path::Path,
    slice, thread,
};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    runtime::Runtime,
    sync::mpsc::{self, Receiver, Sender},
};

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
    send_dataset::send_dataset,
    specs::{server::ServerSpec, worker::WorkerSpec},
};

use crate::{
    OrchErr, Result,
    configs::{AlgorithmConfig, LayerConfig, ModelConfig, Partition},
};

type NetRx = OnoReceiver<OwnedReadHalf>;
type NetTx = OnoSender<OwnedWriteHalf>;

#[derive(Debug, Clone, Copy)]
enum SessionAlgorithm {
    ParameterServer,
    RingAllReduce,
}

/// The result of a completed training session.
///
/// Contains the trained model parameters alongside the model architecture,
/// allowing the weights to be saved to disk without requiring additional context.
#[derive(Debug)]
pub struct TrainedModel {
    /// The flat parameter vector received from the parameter servers.
    pub params: Vec<f32>,
    /// The model architecture used during training.
    pub model: ModelConfig,
    /// The input size of the first layer, derived from the dataset's `x_size`.
    pub input_size: usize,
}

impl TrainedModel {
    /// Returns the weights as a flat slice.
    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// Saves the trained model parameters to a `.safetensors` file.
    ///
    /// Each dense layer produces two tensors named `layer_N.weight` and
    /// `layer_N.bias`, following the PyTorch `state_dict` convention.
    /// The weight tensor has shape `[input_size, output_size]` and the
    /// bias tensor has shape `[output_size]`.
    ///
    /// # Args
    /// * `path` - The output file path (e.g. `"model.safetensors"`).
    ///
    /// # Errors
    /// Returns an `OrchErr` if the file cannot be written or the parameter
    /// buffer does not match the model architecture.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        use safetensors::Dtype;
        use safetensors::tensor::TensorView;

        let mut tensors: Vec<(String, TensorView)> = Vec::new();
        let mut offset = 0;
        let mut prev = self.input_size;

        let params_bytes = unsafe {
            slice::from_raw_parts(self.params.as_ptr() as *const u8, self.params.len() * 4)
        };

        for (i, layer) in self.model.layers.iter().enumerate() {
            let LayerConfig::Dense { output_size, .. } = layer;
            let out = output_size.get();
            let w_count = prev * out;
            let b_count = out;

            let w_bytes = &params_bytes[offset * 4..(offset + w_count) * 4];
            tensors.push((
                format!("layer_{i}.weight"),
                TensorView::new(Dtype::F32, vec![prev, out], w_bytes)
                    .map_err(|e| OrchErr::Io(io::Error::other(e.to_string())))?,
            ));
            offset += w_count;

            let b_bytes = &params_bytes[offset * 4..(offset + b_count) * 4];
            tensors.push((
                format!("layer_{i}.bias"),
                TensorView::new(Dtype::F32, vec![out], b_bytes)
                    .map_err(|e| OrchErr::Io(io::Error::other(e.to_string())))?,
            ));
            offset += b_count;

            prev = out;
        }

        safetensors::tensor::serialize_to_file(
            tensors.iter().map(|(k, v)| (k.as_str(), v.clone())),
            &None,
            path.as_ref(),
        )
        .map_err(|e| OrchErr::Io(io::Error::other(e.to_string())))?;

        info!("model saved to {}", path.as_ref().display());
        Ok(())
    }
}

/// An event produced during a training session.
#[derive(Debug)]
pub enum TrainingEvent {
    /// A worker completed an epoch and reported its losses.
    Loss { worker_id: usize, losses: Vec<f32> },
    /// A worker finished and disconnected.
    WorkerDone(usize),
    /// Training completed and all servers returned the final trained model.
    Complete(TrainedModel),
    /// A worker or server produced an unrecoverable error.
    Error(OrchErr),
}

/// Represents an ongoing training session.
pub struct Session {
    runtime: Runtime,
    servers: Vec<(NetRx, NetTx)>,
    workers: Vec<(NetRx, NetTx)>,
    model: ModelConfig,
    input_size: usize,
    algorithm: SessionAlgorithm,
}

impl Session {
    /// Creates a new session by connecting to all workers and servers.
    ///
    /// # Args
    /// * `workers` - List of (address, spec) pairs for each worker.
    /// * `partitions` - List of dataset partitions for each worker.
    /// * `servers` - List of (address, spec) pairs for each parameter server.
    /// * `model` - The model architecture, kept for post-training serialization.
    /// * `input_size` - The input size of the first layer, derived from the dataset.
    /// * `algorithm` - The distributed training algorithm for this session.
    ///
    /// # Returns
    /// A ready session with all connections established.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or bootstrap message fails.
    pub fn new(
        workers: Vec<(String, WorkerSpec)>,
        partitions: Vec<Partition>,
        servers: Vec<(String, ServerSpec)>,
        model: ModelConfig,
        input_size: usize,
        algorithm: AlgorithmConfig,
    ) -> Result<Self> {
        let (nworkers, nservers) = (workers.len(), servers.len());
        info!("connecting to {nworkers} workers and {nservers} servers");

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        let server_chans = runtime.block_on(Self::create_servers(servers))?;
        debug!("successfully created all servers");

        let worker_chans = runtime.block_on(Self::create_workers(workers, partitions))?;
        debug!("successfully created all workers");

        let algorithm = match algorithm {
            AlgorithmConfig::ParameterServer { .. } => SessionAlgorithm::ParameterServer,
            AlgorithmConfig::RingAllReduce => SessionAlgorithm::RingAllReduce,
        };

        Ok(Self {
            runtime,
            servers: server_chans,
            workers: worker_chans,
            model,
            input_size,
            algorithm,
        })
    }

    async fn worker_listener(id: usize, mut rx: NetRx, _tx: NetTx, tx: Sender<TrainingEvent>) {
        let mut rx_buf = vec![0; 128];

        loop {
            match rx.recv_into(&mut rx_buf).await {
                Ok(Msg::Control(Command::ReportLoss { losses })) => {
                    debug!("worker {id} reported {} losses", losses.len());
                    let event = TrainingEvent::Loss {
                        worker_id: id,
                        losses: losses.into_owned(),
                    };
                    let _ = tx.send(event).await;
                }
                Ok(Msg::Control(Command::Disconnect)) => {
                    info!("worker {id} disconnected");
                    let _ = tx.send(TrainingEvent::WorkerDone(id)).await;
                    return;
                }
                Ok(msg) => {
                    warn!("worker {id}: unexpected message {msg:?}");
                    let event = TrainingEvent::Error(OrchErr::WorkerError {
                        worker_id: id,
                        msg: format!("unexpected message {msg:?}"),
                    });
                    let _ = tx.send(event).await;
                    return;
                }
                Err(e) if is_eof(&e) => {
                    info!("worker {id} closed connection");
                    let _ = tx.send(TrainingEvent::WorkerDone(id)).await;
                    return;
                }
                Err(e) => {
                    error!("worker {id} error: {e}");
                    let event = TrainingEvent::Error(OrchErr::WorkerError {
                        worker_id: id,
                        msg: e.to_string(),
                    });
                    let _ = tx.send(event).await;
                    return;
                }
            }
        }
    }

    /// Consumes `self` and creates an event listener for this training session.
    ///
    /// Spawns a background thread that drives the session, listening to all
    /// workers and parameter servers, and forwards events through the channel.
    ///
    /// # Returns
    /// A receiver that yields `TrainingEvent`s as training progresses.
    pub fn event_listener(self) -> Receiver<TrainingEvent> {
        let (tx, rx) = mpsc::channel(256);
        let model = self.model;
        let input_size = self.input_size;
        let algorithm = self.algorithm;

        thread::spawn(move || {
            self.runtime.block_on(async move {
                let futs = self.workers.into_iter().enumerate().map(|(i, (wrx, wtx))| {
                    tokio::spawn(Self::worker_listener(i, wrx, wtx, tx.clone()))
                });

                future::join_all(futs).await;

                match algorithm {
                    SessionAlgorithm::ParameterServer => {
                        debug!("all workers done, reading final params from all servers");

                        let mut model_params: Vec<f32> = Vec::new();
                        let mut rx_buf = vec![0; 1024];

                        for (i, mut srx) in self.servers.into_iter().map(|(rx, _)| rx).enumerate() {
                            match srx.recv_into(&mut rx_buf).await {
                                Ok(Msg::Data(Payload::Params(params))) => {
                                    model_params.extend_from_slice(params);
                                }
                                Ok(msg) => {
                                    let err = OrchErr::ServerError(format!(
                                        "unexpected message from server {i}: {msg:?}"
                                    ));
                                    let _ = tx.send(TrainingEvent::Error(err)).await;
                                    return;
                                }
                                Err(e) => {
                                    let err = OrchErr::ServerError(format!(
                                        "unexpected error from server {i}: {e}"
                                    ));
                                    let _ = tx.send(TrainingEvent::Error(err)).await;
                                    return;
                                }
                            }
                        }

                        info!("received {} total parameters", model_params.len());

                        let trained = TrainedModel {
                            params: model_params,
                            model,
                            input_size,
                        };

                        let _ = tx.send(TrainingEvent::Complete(trained)).await;
                    }
                    SessionAlgorithm::RingAllReduce => {
                        let err = OrchErr::Unsupported(
                            "ring all-reduce session finalization is not implemented yet".into(),
                        );
                        let _ = tx.send(TrainingEvent::Error(err)).await;
                    }
                }
            });
        });

        rx
    }

    /// Connects to all parameter servers and sends each its bootstrap spec.
    ///
    /// # Args
    /// * `servers` - List of (address, spec) pairs for each parameter server.
    ///
    /// # Returns
    /// A list of open (receiver, sender) channel pairs, one per server.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or send fails.
    async fn create_servers(servers: Vec<(String, ServerSpec)>) -> Result<Vec<(NetRx, NetTx)>> {
        let mut channels = Vec::with_capacity(servers.len());

        for (addr, spec) in servers {
            let (rx, mut tx) = Self::open_channel(&addr)
                .await
                .map_err(|source| OrchErr::ConnectionFailed { addr, source })?;

            tx.send(&Msg::Control(Command::CreateServer(spec))).await?;
            channels.push((rx, tx));
        }

        Ok(channels)
    }

    /// Connects to all workers, sends each its bootstrap spec and dataset partition.
    ///
    /// # Args
    /// * `workers` - List of (address, spec) pairs for each worker.
    /// * `partitions` - List of dataset partitions for each worker.
    ///
    /// # Returns
    /// A list of open (receiver, sender) channel pairs, one per worker.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or send fails.
    async fn create_workers<'a>(
        workers: Vec<(String, WorkerSpec)>,
        partitions: Vec<Partition<'a>>,
    ) -> Result<Vec<(NetRx, NetTx)>> {
        const CHUNK_SIZE: usize = 8192;

        let futs =
            workers
                .into_iter()
                .zip(partitions)
                .map(|((addr, spec), partition)| async move {
                    debug!("connecting to worker at {addr}");

                    let (rx, mut tx) = Self::open_channel(&addr)
                        .await
                        .map_err(|source| OrchErr::ConnectionFailed { addr, source })?;

                    tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

                    match partition {
                        Partition::Local { path, offset, size } => {
                            let mut fd = File::open(path).await?;
                            fd.seek(io::SeekFrom::Start(offset)).await?;
                            let mut fd = fd.take(size);
                            send_dataset(&mut fd, CHUNK_SIZE, &mut tx).await?;
                        }
                        Partition::Inline { data } => {
                            let bytes: &[u8] = bytemuck::cast_slice(data);
                            let mut cursor = Cursor::new(bytes);
                            send_dataset(&mut cursor, CHUNK_SIZE, &mut tx).await?;
                        }
                    }

                    Ok::<_, OrchErr>((rx, tx))
                });

        let channels = future::try_join_all(futs).await?;
        Ok(channels)
    }

    /// Opens a TCP channel to the given address.
    ///
    /// # Args
    /// * `addr` - The socket address to connect to.
    ///
    /// # Returns
    /// A (receiver, sender) channel pair.
    ///
    /// # Errors
    /// Returns an `io::Error` if the connection fails.
    async fn open_channel(addr: &str) -> io::Result<(NetRx, NetTx)> {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        Ok(comms::channel(rx, tx))
    }
}

fn is_eof(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset
    ) || e.to_string().contains("early eof")
}
