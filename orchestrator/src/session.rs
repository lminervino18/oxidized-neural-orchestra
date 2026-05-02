use std::{
    collections::HashMap,
    io::{self, Cursor},
    path::{Path, PathBuf},
    slice, thread,
};

use comms::{
    Connector, NetRtp, ParamServerHandle, TransportLayer, WorkerEvent, WorkerHandle,
    protocol::Entity,
    specs::{
        server::ServerSpec,
        worker::{AlgorithmSpec, WorkerSpec},
    },
};
use futures::future;
use log::{debug, error, info, warn};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    runtime::{Builder, Runtime},
    sync::mpsc::{self, Receiver, Sender},
};

use crate::{
    OrchErr, Result,
    configs::{EarlyStoppingConfig, LayerConfig, ModelConfig, Partition},
};

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
        use safetensors::{Dtype, tensor::TensorView};

        let mut tensors: Vec<(String, TensorView)> = Vec::new();
        let mut offset = 0;
        let mut prev = self.input_size;

        // SAFETY: we cast &[f32] to &[u8] for safetensors — f32 is always 4 bytes,
        // alignment is valid, and the slice is live for the duration of this function.
        let params_bytes = unsafe {
            slice::from_raw_parts(self.params.as_ptr() as *const u8, self.params.len() * 4)
        };

        for (i, layer) in self.model.layers.iter().enumerate() {
            let (w_count, b_count, w_shape, b_shape, out) = match layer {
                LayerConfig::Dense { output_size, .. } => {
                    let out = output_size.get();
                    let w_count = prev * out;
                    let b_count = out;

                    (w_count, b_count, vec![prev, out], vec![out], out)
                }
                LayerConfig::Conv {
                    input_dim,
                    kernel_dim,
                    stride,
                    padding,
                    ..
                } => {
                    let input_height = input_dim.1.get();
                    let input_width = input_dim.2.get();
                    let (filters, in_channels, kernel_size) =
                        (kernel_dim.0.get(), kernel_dim.1.get(), kernel_dim.2.get());
                    let stride = stride.get();

                    let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
                    let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

                    let w_count = filters * in_channels * kernel_size * kernel_size;
                    let b_count = filters;
                    let out = output_height * output_width * filters;

                    (
                        w_count,
                        b_count,
                        vec![filters, in_channels, kernel_size, kernel_size],
                        vec![filters],
                        out,
                    )
                }
            };

            let w_bytes = &params_bytes[offset * 4..(offset + w_count) * 4];
            tensors.push((
                format!("layer_{i}.weight"),
                TensorView::new(Dtype::F32, w_shape, w_bytes)
                    .map_err(|e| OrchErr::Io(io::Error::other(e.to_string())))?,
            ));
            offset += w_count;

            let b_bytes = &params_bytes[offset * 4..(offset + b_count) * 4];
            tensors.push((
                format!("layer_{i}.bias"),
                TensorView::new(Dtype::F32, b_shape, b_bytes)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopReason {
    #[default]
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
    Complete {
        model: TrainedModel,
        reason: StopReason,
    },
    /// The parameters of the all reduce worker.
    Params(Vec<f32>),
    /// A worker or server produced an unrecoverable error.
    Error(OrchErr),
}

#[derive(Debug)]
enum WorkerHandleRequest {
    PullParams,
    Stop,
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
    servers: Vec<ParamServerHandle<NetRtp>>,
    workers: Vec<WorkerHandle<NetRtp>>,
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
    /// * `connector` - The underlying entity connector.
    /// * `model` - The model architecture, kept for post-training serialization.
    /// * `input_size` - The input size of the first layer, derived from the dataset.
    ///
    /// # Returns
    /// A ready session with all connections established.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or bootstrap message fails.
    pub fn new<F>(
        workers: Vec<(String, WorkerSpec)>,
        partitions: Vec<Partition>,
        servers: Vec<(String, ServerSpec)>,
        connector: Connector<OwnedReadHalf, OwnedWriteHalf, NetRtp, F>,
        model: ModelConfig,
        input_size: usize,
        early_stopping: Option<EarlyStoppingConfig>,
    ) -> Result<Self>
    where
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> NetRtp,
    {
        let algorithm = workers
            .first()
            .map(|(_, spec)| spec.algorithm.clone())
            .ok_or_else(|| OrchErr::InvalidConfig("at least one worker is required".into()))?;

        let (nworkers, nservers) = (workers.len(), servers.len());
        info!("connecting to {nworkers} workers and {nservers} servers");

        let runtime = Builder::new_multi_thread().enable_all().build()?;

        let server_chans = runtime.block_on(Self::create_servers(servers, &connector))?;
        debug!("successfully created all servers");

        let worker_chans =
            runtime.block_on(Self::create_workers(workers, partitions, &connector))?;
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

    async fn worker_listener(
        id: usize,
        mut worker_handle: WorkerHandle<NetRtp>,
        mut rx_stopper: Receiver<WorkerHandleRequest>,
        tx: Sender<TrainingEvent>,
    ) {
        let mut stopping = false;
        loop {
            tokio::select! {
                req = rx_stopper.recv() => {
                    match req {
                        Some(WorkerHandleRequest::Stop) if !stopping => {
                            stopping = true;
                            if let Err(e) = worker_handle.stop().await {
                                error!("worker {id}: failed to send stop command: {e}");

                                let event = TrainingEvent::Error(OrchErr::WorkerError {
                                    worker_id: id,
                                    event: format!("failed to send stop command: {e}"),
                                });

                                let _ = tx.send(event).await;
                                return;
                            }
                        },
                        Some(WorkerHandleRequest::PullParams) => {
                            match worker_handle.pull_params().await {
                                Ok(params) => {let _ = tx.send(TrainingEvent::Params(params.to_vec())).await;},
                                Err(e) => {
                                    error!("worker {id}: failed to pull params command: {e}");

                                    let event = TrainingEvent::Error(OrchErr::WorkerError {
                                        worker_id: id,
                                        event: format!("failed to pull params command: {e}"),
                                    });

                                    let _ = tx.send(event).await;
                                    return;
                                },
                            }
                        }
                        _ => {}
                    }
                },
                event = worker_handle.recv_event() => match event {
                    Ok(WorkerEvent::Loss(losses)) => {
                        debug!("worker {id} reported {} losses", losses.len());

                        let event = TrainingEvent::Loss {
                            worker_id: id,
                            losses,
                        };

                        let _ = tx.send(event).await;
                    }
                    Ok(WorkerEvent::Disconnect) => {
                        info!("worker {id} disconnected");
                        let _ = tx.send(TrainingEvent::WorkerDone(id)).await;
                        return;
                    }
                    Ok(event) => {
                        warn!("worker {id}: unexpected event {event:?}");

                        let event = TrainingEvent::Error(OrchErr::WorkerError {
                            worker_id: id,
                            event: format!("unexpected event {event:?}"),
                        });

                        let _ = tx.send(event).await;
                        return;
                    }
                    Err(e) if is_eof(&e) => {
                        info!("worker {id}'s connection closed");
                        let _ = tx.send(TrainingEvent::WorkerDone(id)).await;
                        return;
                    }
                    Err(e) => {
                        error!("worker {id} error: {e}");

                        let event = TrainingEvent::Error(OrchErr::WorkerError {
                            worker_id: id,
                            event: e.to_string(),
                        });

                        let _ = tx.send(event).await;
                        return;
                    }
                }
            }
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
                let (internal_tx, mut internal_rx) = mpsc::channel(256);
                let n_workers = self.workers.len();

                let mut wk_txs = Self::spawn_worker_listeners(self.workers, internal_tx);

                let Some(reason) = Self::run_training_loop(
                    &mut cancel_rx,
                    &mut internal_rx,
                    &mut wk_txs,
                    n_workers,
                    &early_stopping,
                    &event_tx,
                )
                .await
                else {
                    return;
                };

                let params = match algorithm {
                    AlgorithmSpec::ParameterServer { .. } => {
                        Self::finalize_parameter_server(self.servers, &event_tx).await
                    }
                    AlgorithmSpec::AllReduce { .. } => {
                        let mut i = 0;
                        loop {
                            let _ = wk_txs[i].send(WorkerHandleRequest::PullParams).await;
                            if let Some(TrainingEvent::Params(params)) = internal_rx.recv().await {
                                break params.to_vec();
                            }

                            i = (i + 1) % wk_txs.len();
                        }
                    }
                };

                info!("received {} total parameters", params.len());

                let trained = TrainedModel {
                    params,
                    model,
                    input_size,
                };

                let _ = event_tx
                    .send(TrainingEvent::Complete {
                        model: trained,
                        reason,
                    })
                    .await;
            });
        });

        event_rx
    }

    fn spawn_worker_listeners(
        workers: Vec<WorkerHandle<NetRtp>>,
        internal_tx: Sender<TrainingEvent>,
    ) -> Vec<Sender<WorkerHandleRequest>> {
        let mut wk_txs = Vec::with_capacity(workers.len());

        for (i, worker_handle) in workers.into_iter().enumerate() {
            let (wk_tx, wk_rx) = mpsc::channel(256);
            wk_txs.push(wk_tx);
            tokio::spawn(Self::worker_listener(
                i,
                worker_handle,
                wk_rx,
                internal_tx.clone(),
            ));
        }

        wk_txs
    }

    async fn run_training_loop(
        cancel_rx: &mut mpsc::Receiver<()>,
        internal_rx: &mut mpsc::Receiver<TrainingEvent>,
        wk_txs: &mut Vec<Sender<WorkerHandleRequest>>,
        n_workers: usize,
        early_stopping: &Option<EarlyStoppingConfig>,
        event_tx: &Sender<TrainingEvent>,
    ) -> Option<StopReason> {
        let mut tracker = ConvergenceTracker::new(n_workers);
        let mut stop_reason: Option<StopReason> = None;
        let mut workers_done = 0;

        loop {
            tokio::select! {
                biased;
                _ = cancel_rx.recv(), if stop_reason.is_none() => {
                    info!("manual stop requested");
                    stop_reason = Some(StopReason::ManualStop);

                    for tx in wk_txs.iter_mut() { let _ = tx.send(WorkerHandleRequest::Stop).await; }
                }
                evt = internal_rx.recv() => {
                    let Some(event) = evt else {
                        break;
                    };

                    match event {
                        TrainingEvent::WorkerDone(id) => {
                            workers_done += 1;
                            let _ = event_tx.send(TrainingEvent::WorkerDone(id)).await;
                            if workers_done == n_workers {  break; }
                        }
                        TrainingEvent::Loss { worker_id, losses } => {
                            if stop_reason.is_none() {
                                if let Some(cfg) = early_stopping {
                                    if let Some(sig) = tracker.record(worker_id, &losses) {
                                        if cfg.is_converged(sig.prev, sig.curr) {
                                            info!(
                                                "early stopping triggered (prev={:.6}, curr={:.6})",
                                                sig.prev, sig.curr
                                            );

                                            stop_reason = Some(StopReason::EarlyStopping);
                                            for tx in wk_txs.iter_mut() {
                                                let _ = tx.send(WorkerHandleRequest::Stop).await;
                                            }
                                        }
                                    }
                                }
                            }

                            let _ = event_tx.send(TrainingEvent::Loss { worker_id, losses }).await;
                        }
                        TrainingEvent::Error(e) => {
                            let _ = event_tx.send(TrainingEvent::Error(e)).await;
                            return None;
                        }
                        other => { let _ = event_tx.send(other).await; }
                    }
                }
            }
        }

        Some(stop_reason.unwrap_or_default())
    }

    async fn finalize_parameter_server(
        servers: Vec<ParamServerHandle<NetRtp>>,
        event_tx: &Sender<TrainingEvent>,
    ) -> Vec<f32> {
        debug!("all workers done, reading final params from all servers");
        let mut model_params: Vec<f32> = Vec::new();

        for (i, mut server_handle) in servers.into_iter().enumerate() {
            match server_handle.pull_params().await {
                Ok(params) => {
                    model_params.extend_from_slice(params);
                    if let Err(e) = server_handle.disconnect().await {
                        error!("Failed to disconnect server {i}: {e}");
                    }
                }
                Err(e) => {
                    let err =
                        OrchErr::ServerError(format!("unexpected error from server {i}: {e}"));
                    let _ = event_tx.send(TrainingEvent::Error(err)).await;
                }
            }
        }

        model_params
    }

    /// Connects to all nodes and bootstraps each as a parameter server.
    async fn create_servers<F>(
        servers: Vec<(String, ServerSpec)>,
        connector: &Connector<OwnedReadHalf, OwnedWriteHalf, NetRtp, F>,
    ) -> Result<Vec<ParamServerHandle<NetRtp>>>
    where
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> NetRtp,
    {
        let mut handles = Vec::with_capacity(servers.len());

        for (i, (addr, spec)) in servers.into_iter().enumerate() {
            let stream =
                TcpStream::connect(&addr)
                    .await
                    .map_err(|e| OrchErr::ConnectionFailed {
                        addr: addr.clone(),
                        source: e,
                    })?;
            let (rx, tx) = stream.into_split();
            let node_handle = connector
                .connect_node(i, rx, tx, Entity::Orchestrator)
                .await
                .map_err(|e| OrchErr::ConnectionFailed { addr, source: e })?;

            let server_handle = node_handle.create_server(spec).await?;
            handles.push(server_handle);
        }

        Ok(handles)
    }

    /// Connects to all nodes, bootstraps each as a worker, and sends its dataset partition.
    async fn create_workers<'a, F>(
        workers: Vec<(String, WorkerSpec)>,
        partitions: Vec<Partition<'a>>,
        connector: &Connector<OwnedReadHalf, OwnedWriteHalf, NetRtp, F>,
    ) -> Result<Vec<WorkerHandle<NetRtp>>>
    where
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> NetRtp,
    {
        const CHUNK_SIZE: usize = 8192;

        let futs = workers.into_iter().zip(partitions).enumerate().map(
            |(i, ((addr, spec), partition))| async move {
                debug!("connecting to worker at {addr}");

                let stream =
                    TcpStream::connect(&addr)
                        .await
                        .map_err(|e| OrchErr::ConnectionFailed {
                            addr: addr.clone(),
                            source: e,
                        })?;
                let (rx, tx) = stream.into_split();
                let node_handle = connector
                    .connect_node(i, rx, tx, Entity::Orchestrator)
                    .await
                    .map_err(|e| OrchErr::ConnectionFailed { addr, source: e })?;

                let mut worker_handle = node_handle.create_worker(spec).await?;
                Self::send_partition(partition, CHUNK_SIZE, &mut worker_handle).await?;

                Ok::<_, OrchErr>(worker_handle)
            },
        );

        future::try_join_all(futs).await
    }

    async fn send_partition<'a, T: TransportLayer>(
        partition: Partition<'a>,
        chunk_size: usize,
        worker_handle: &mut WorkerHandle<T>,
    ) -> Result<()> {
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
                    chunk_size,
                    worker_handle,
                )
                .await
            }
            Partition::Inline { samples, labels } => {
                Self::send_inline_partition(samples, labels, chunk_size, worker_handle).await
            }
        }
    }

    async fn send_local_partition<T>(
        samples_path: &PathBuf,
        labels_path: &PathBuf,
        samples_offset: u64,
        labels_offset: u64,
        samples_size: u64,
        labels_size: u64,
        chunk_size: usize,
        worker_handle: &mut WorkerHandle<T>,
    ) -> Result<()>
    where
        T: TransportLayer,
    {
        let mut samples_fd = File::open(samples_path).await?;
        let mut labels_fd = File::open(labels_path).await?;

        samples_fd.seek(io::SeekFrom::Start(samples_offset)).await?;
        labels_fd.seek(io::SeekFrom::Start(labels_offset)).await?;
        let mut samples_fd = samples_fd.take(samples_size);
        let mut labels_fd = labels_fd.take(labels_size);

        worker_handle
            .push_dataset(&mut samples_fd, &mut labels_fd, chunk_size)
            .await?;

        Ok(())
    }

    async fn send_inline_partition<T>(
        samples: &[f32],
        labels: &[f32],
        chunk_size: usize,
        worker_handle: &mut WorkerHandle<T>,
    ) -> Result<()>
    where
        T: TransportLayer,
    {
        let sample_bytes: &[u8] = bytemuck::cast_slice(samples);
        let label_bytes: &[u8] = bytemuck::cast_slice(labels);
        let mut samples_cursor = Cursor::new(sample_bytes);
        let mut labels_cursor = Cursor::new(label_bytes);

        worker_handle
            .push_dataset(&mut samples_cursor, &mut labels_cursor, chunk_size)
            .await?;

        Ok(())
    }
}

fn is_eof(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset
    ) || e.to_string().contains("early eof")
}
