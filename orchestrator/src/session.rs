use futures::future;
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::path::PathBuf;
use std::{
    io::{self, Cursor},
    path::Path,
    slice, thread,
};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWrite},
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
    specs::{
        server::ServerSpec,
        worker::{AlgorithmSpec, WorkerSpec},
    },
};

use crate::{
    OrchErr, Result,
    configs::{EarlyStoppingConfig, LayerConfig, ModelConfig, Partition},
};

type NetRx = OnoReceiver<OwnedReadHalf>;
type NetTx = OnoSender<OwnedWriteHalf>;

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

        // SAFETY: we cast &[f32] to &[u8] for safetensors — f32 is always 4 bytes,
        // alignment is valid, and the slice is live for the duration of this function.
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

/// Why a training session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    MaxEpochsReached,
    EarlyStopping,
    ManualStop,
}

/// A handle that lets any caller request an early stop of an ongoing training session.
pub struct CancelHandle(mpsc::Sender<()>);

impl CancelHandle {
    /// Creates a matched `(CancelHandle, Receiver)` pair.
    ///
    /// The caller retains the `CancelHandle` and passes the `Receiver` to
    /// `Session::event_listener`. Calling `stop()` on the handle signals
    /// the session to stop at the next epoch boundary.
    pub fn pair() -> (Self, mpsc::Receiver<()>) {
        let (tx, rx) = mpsc::channel(1);
        (Self(tx), rx)
    }

    pub fn stop(&self) {
        let _ = self.0.try_send(());
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
    Complete { model: TrainedModel, reason: StopReason },
    /// A worker or server produced an unrecoverable error.
    Error(OrchErr),
}

struct SyncRoundSignal {
    prev: f32,
    curr: f32,
}

struct ConvergenceTracker {
    n_workers: usize,
    pending: HashMap<usize, f32>,
    prev_avg: Option<f32>,
}

impl ConvergenceTracker {
    fn new(n_workers: usize) -> Self {
        Self {
            n_workers,
            pending: HashMap::new(),
            prev_avg: None,
        }
    }

    fn record(&mut self, worker_id: usize, losses: &[f32]) -> Option<SyncRoundSignal> {
        let last = *losses.last()?;
        self.pending.insert(worker_id, last);

        if self.pending.len() < self.n_workers {
            return None;
        }

        let curr = self.pending.values().sum::<f32>() / self.n_workers as f32;
        self.pending.clear();

        let signal = self.prev_avg.map(|prev| SyncRoundSignal { prev, curr });
        self.prev_avg = Some(curr);
        signal
    }
}

/// Represents an ongoing training session.
pub struct Session {
    runtime: Runtime,
    servers: Vec<(NetRx, NetTx)>,
    workers: Vec<(NetRx, NetTx)>,
    model: ModelConfig,
    input_size: usize,
    algorithm: AlgorithmSpec,
    early_stopping: Option<EarlyStoppingConfig>,
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
        early_stopping: Option<EarlyStoppingConfig>,
    ) -> Result<Self> {
        let algorithm = workers
            .first()
            .map(|(_, spec)| spec.algorithm.clone())
            .ok_or_else(|| OrchErr::InvalidConfig("at least one worker is required".into()))?;

        let (nworkers, nservers) = (workers.len(), servers.len());
        info!("connecting to {nworkers} workers and {nservers} servers");

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        let (server_chans, session_ids) = runtime.block_on(Self::create_servers(servers))?;
        debug!("successfully created all servers");

        let workers = workers
            .into_iter()
            .map(|(addr, mut spec)| {
                if let AlgorithmSpec::ParameterServer {
                    ref mut server_session_ids,
                    ..
                } = spec.algorithm
                {
                    *server_session_ids = session_ids.clone();
                }
                (addr, spec)
            })
            .collect::<Vec<_>>();

        let worker_chans = runtime.block_on(Self::create_workers(workers, partitions))?;
        debug!("successfully created all workers");

        Ok(Self {
            runtime,
            servers: server_chans,
            workers: worker_chans,
            model,
            input_size,
            algorithm,
            early_stopping,
        })
    }

    async fn worker_listener(id: usize, mut rx: NetRx, tx: Sender<TrainingEvent>) {
        loop {
            match rx.recv().await {
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

    async fn broadcast_stop(worker_txs: &mut [NetTx]) {
        for tx in worker_txs.iter_mut() {
            let _ = tx.send(&Msg::Control(Command::StopAfterEpoch)).await;
        }
    }

    /// Consumes `self` and creates an event listener for this training session.
    ///
    /// Spawns a background thread that drives the session. The `cancel_rx` must come
    /// from a `CancelHandle::pair()` call; the caller retains the `CancelHandle` to
    /// request an orderly stop at any time.
    ///
    /// # Returns
    /// A receiver that yields `TrainingEvent`s as training progresses.
    pub fn event_listener(self, mut cancel_rx: mpsc::Receiver<()>) -> Receiver<TrainingEvent> {
        let (event_tx, event_rx) = mpsc::channel(256);

        let model = self.model;
        let input_size = self.input_size;
        let algorithm = self.algorithm;
        let early_stopping = self.early_stopping;

        thread::spawn(move || {
            self.runtime.block_on(async move {
                let (internal_tx, mut internal_rx) = mpsc::channel::<TrainingEvent>(256);

                let n_workers = self.workers.len();
                let (worker_rxs, mut worker_txs): (Vec<_>, Vec<_>) =
                    self.workers.into_iter().unzip();

                for (i, wrx) in worker_rxs.into_iter().enumerate() {
                    tokio::spawn(Self::worker_listener(i, wrx, internal_tx.clone()));
                }
                drop(internal_tx);

                let mut tracker = ConvergenceTracker::new(n_workers);
                let mut stop_reason: Option<StopReason> = None;
                let mut workers_done = 0;

                loop {
                    tokio::select! {
                        biased;
                        _ = cancel_rx.recv(), if stop_reason.is_none() => {
                            info!("manual stop requested");
                            stop_reason = Some(StopReason::ManualStop);
                            Self::broadcast_stop(&mut worker_txs).await;
                        }
                        evt = internal_rx.recv() => {
                            match evt {
                                None => break,
                                Some(TrainingEvent::WorkerDone(id)) => {
                                    workers_done += 1;
                                    let _ = event_tx.send(TrainingEvent::WorkerDone(id)).await;
                                    if workers_done == n_workers {
                                        break;
                                    }
                                }
                                Some(TrainingEvent::Loss { worker_id, losses }) => {
                                    if stop_reason.is_none() {
                                        if let Some(cfg) = &early_stopping {
                                            if let Some(sig) = tracker.record(worker_id, &losses) {
                                                if cfg.is_converged(sig.prev, sig.curr) {
                                                    info!(
                                                        "early stopping triggered (prev={:.6}, curr={:.6})",
                                                        sig.prev, sig.curr
                                                    );
                                                    stop_reason = Some(StopReason::EarlyStopping);
                                                    Self::broadcast_stop(&mut worker_txs).await;
                                                }
                                            }
                                        }
                                    }
                                    let _ = event_tx
                                        .send(TrainingEvent::Loss { worker_id, losses })
                                        .await;
                                }
                                Some(TrainingEvent::Error(e)) => {
                                    let _ = event_tx.send(TrainingEvent::Error(e)).await;
                                    return;
                                }
                                Some(other) => {
                                    let _ = event_tx.send(other).await;
                                }
                            }
                        }
                    }
                }

                let reason = stop_reason.unwrap_or(StopReason::MaxEpochsReached);

                match algorithm {
                    AlgorithmSpec::ParameterServer { .. } => {
                        debug!("all workers done, reading final params from all servers");

                        let mut model_params: Vec<f32> = Vec::new();

                        for (i, mut srx) in
                            self.servers.into_iter().map(|(rx, _)| rx).enumerate()
                        {
                            match srx.recv().await {
                                Ok(Msg::Data(Payload::Params(params))) => {
                                    model_params.extend_from_slice(params);
                                }
                                Ok(msg) => {
                                    let err = OrchErr::ServerError(format!(
                                        "unexpected message from server {i}: {msg:?}"
                                    ));
                                    let _ = event_tx.send(TrainingEvent::Error(err)).await;
                                    return;
                                }
                                Err(e) => {
                                    let err = OrchErr::ServerError(format!(
                                        "unexpected error from server {i}: {e}"
                                    ));
                                    let _ = event_tx.send(TrainingEvent::Error(err)).await;
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

                        let _ = event_tx
                            .send(TrainingEvent::Complete {
                                model: trained,
                                reason,
                            })
                            .await;
                    }
                    AlgorithmSpec::AllReduce { .. } => {
                        let err = OrchErr::Unsupported(
                            "all-reduce session finalization is not implemented yet".into(),
                        );
                        let _ = event_tx.send(TrainingEvent::Error(err)).await;
                    }
                }
            });
        });

        event_rx
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
    async fn create_servers(
        servers: Vec<(String, ServerSpec)>,
    ) -> Result<(Vec<(NetRx, NetTx)>, Vec<u64>)> {
        let mut channels = Vec::with_capacity(servers.len());
        let mut session_ids = Vec::with_capacity(servers.len());

        for (addr, spec) in servers {
            let (mut rx, mut tx) = Self::open_channel(&addr)
                .await
                .map_err(|source| OrchErr::ConnectionFailed { addr, source })?;

            tx.send(&Msg::Control(Command::CreateServer(spec))).await?;

            match rx.recv().await? {
                Msg::Control(Command::ServerReady { session_id }) => {
                    session_ids.push(session_id);
                }
                msg => {
                    return Err(OrchErr::ServerError(format!(
                        "expected ServerReady, got {msg:?}"
                    )));
                }
            }

            channels.push((rx, tx));
        }

        Ok((channels, session_ids))
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
                        Partition::Local {
                            samples_path,
                            labels_path,
                            samples_offset,
                            labels_offset,
                            samples_size,
                            labels_size,
                        } => {
                            Self::send_local_partition(
                                samples_path,
                                labels_path,
                                samples_offset,
                                labels_offset,
                                samples_size,
                                labels_size,
                                CHUNK_SIZE,
                                &mut tx,
                            )
                            .await?;
                        }
                        Partition::Inline { samples, labels } => {
                            Self::send_inline_partition(samples, labels, CHUNK_SIZE, &mut tx)
                                .await?;
                        }
                    }

                    Ok::<_, OrchErr>((rx, tx))
                });

        let channels = future::try_join_all(futs).await?;
        Ok(channels)
    }

    async fn send_local_partition<W>(
        samples_path: &PathBuf,
        labels_path: &PathBuf,
        samples_offset: u64,
        labels_offset: u64,
        samples_size: u64,
        labels_size: u64,
        chunk_size: usize,
        tx: &mut OnoSender<W>,
    ) -> Result<()>
    where
        W: AsyncWrite + Unpin,
    {
        let mut samples_fd = File::open(samples_path).await?;
        let mut labels_fd = File::open(labels_path).await?;

        samples_fd.seek(io::SeekFrom::Start(samples_offset)).await?;
        labels_fd.seek(io::SeekFrom::Start(labels_offset)).await?;
        let mut samples_fd = samples_fd.take(samples_size);
        let mut labels_fd = labels_fd.take(labels_size);

        send_dataset(&mut samples_fd, &mut labels_fd, chunk_size, tx).await?;

        Ok(())
    }

    async fn send_inline_partition<W>(
        samples: &[f32],
        labels: &[f32],
        chunk_size: usize,
        tx: &mut OnoSender<W>,
    ) -> Result<()>
    where
        W: AsyncWrite + Unpin,
    {
        let sample_bytes: &[u8] = bytemuck::cast_slice(samples);
        let label_bytes: &[u8] = bytemuck::cast_slice(labels);
        let mut samples_cursor = Cursor::new(sample_bytes);
        let mut labels_cursor = Cursor::new(label_bytes);

        send_dataset(&mut samples_cursor, &mut labels_cursor, chunk_size, tx).await?;

        Ok(())
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
