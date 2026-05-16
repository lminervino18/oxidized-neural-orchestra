use std::{
    collections::HashMap,
    io::{self, SeekFrom},
    path::Path,
    thread,
};

use comms::{
    Connector, NetRtp, ParamServerHandle, TransportLayer, WorkerEvent, WorkerHandle,
    protocol::Entity,
    share_dataset,
    specs::{server::ServerSpec, worker::WorkerSpec},
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

use super::TrainedModel;
use crate::{
    OrchErr, Result, TrainingEvent,
    configs::{AlgorithmConfig, EarlyStoppingConfig, ModelConfig, Partition},
    sessions::WorkerRequest,
};

/// Why a training session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopReason {
    #[default]
    MaxEpochsReached,
    EarlyStopping,
    ManualStop,
}

/// A handle that lets any caller request an early stop of an ongoing training session.
pub struct CancelHandle(Sender<()>);

impl CancelHandle {
    /// Creates a matched `(CancelHandle, Receiver)` pair.
    ///
    /// The caller retains the `CancelHandle` and passes the `Receiver` to
    /// `Session::event_listener`. Calling `stop()` on the handle signals
    /// the session to stop at the next epoch boundary.
    pub fn pair() -> (Self, Receiver<()>) {
        let (tx, rx) = mpsc::channel(1);
        (Self(tx), rx)
    }

    pub fn stop(&self) {
        let _ = self.0.try_send(());
    }
}

struct SyncRoundSignal {
    prev: f64,
    curr: f64,
}

struct ConvergenceTracker {
    n_workers: usize,
    pending: HashMap<usize, f64>,
    prev_avg: Option<f64>,
}

impl ConvergenceTracker {
    fn new(n_workers: usize) -> Self {
        Self {
            n_workers,
            pending: HashMap::new(),
            prev_avg: None,
        }
    }

    fn record(&mut self, worker_id: usize, losses: &[f64]) -> Option<SyncRoundSignal> {
        let last = *losses.last()?;
        self.pending.insert(worker_id, last);

        if self.pending.len() < self.n_workers {
            return None;
        }

        let pending_sum: f64 = self.pending.values().sum();
        let curr = pending_sum / self.n_workers as f64;
        self.pending.clear();

        let signal = self.prev_avg.map(|prev| SyncRoundSignal { prev, curr });
        self.prev_avg = Some(curr);
        signal
    }
}

/// An ongoing training session.
pub struct Session {
    runtime: Runtime,
    workers: Vec<WorkerHandle<NetRtp>>,
    servers: Vec<ParamServerHandle<NetRtp>>,
    model: ModelConfig,
    algorithm: AlgorithmConfig,
    input_size: usize,
    early_stopping: Option<EarlyStoppingConfig>,
}

impl Session {
    /// Creates a new session by connecting to all workers and servers.
    ///
    /// # Args
    /// * `workers` - The workers' network addresses, specifications and dataset partitions.
    /// * `servers` - The servers' network addresses and specifications.
    /// * `connector` - The network connector.
    /// * `model` - The model's architecture configuration.
    /// * `training` - The model's training configuration.
    /// * `input_size` - The model's input size.
    ///
    /// # Returns
    /// A ready session with all connections established.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or bootstrap message fails.
    pub fn new<F>(
        workers: Vec<(String, WorkerSpec, Partition<'_>)>,
        servers: Vec<(String, ServerSpec)>,
        connector: Connector<OwnedReadHalf, OwnedWriteHalf, NetRtp, F>,
        model: ModelConfig,
        algorithm: AlgorithmConfig,
        input_size: usize,
        early_stopping: Option<EarlyStoppingConfig>,
    ) -> Result<Self>
    where
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> NetRtp,
    {
        let runtime = Builder::new_multi_thread().enable_all().build()?;
        let (nworkers, nservers) = (workers.len(), servers.len());

        let server_handles = match algorithm {
            AlgorithmConfig::ParameterServer { .. } => {
                info!("connecting to {nservers} servers");
                let server_handles = runtime.block_on(Self::create_servers(servers, &connector))?;
                debug!("successfully created servers");
                server_handles
            }
            _ => Vec::new(),
        };

        info!("connecting to {nworkers} workers");
        let worker_handles = runtime.block_on(Self::create_workers(workers, &connector))?;
        debug!("successfully created workers");

        Ok(Self {
            runtime,
            workers: worker_handles,
            servers: server_handles,
            model,
            algorithm,
            input_size,
            early_stopping,
        })
    }

    async fn worker_listener(
        id: usize,
        mut worker_handle: WorkerHandle<NetRtp>,
        mut wk_rx: Receiver<WorkerRequest>,
        tx: Sender<TrainingEvent>,
    ) {
        let mut stopping = false;
        loop {
            tokio::select! {
                req = wk_rx.recv() => match req {
                    Some(WorkerRequest::Stop) if stopping => {}
                    Some(WorkerRequest::Stop) => {
                        stopping = true;

                        if let Err(e) = worker_handle.stop().await {
                            error!("worker {id}: failed to send stop command: {e}");

                            let event = TrainingEvent::Error(OrchErr::WorkerError {
                                id,
                                details: format!("failed to send stop command: {e}"),
                            });

                            let _ = tx.send(event).await;
                            return;
                        }
                    },
                    Some(WorkerRequest::PullParams) => {
                        match worker_handle.pull_params().await {
                            Ok(params) => {
                                let _ = tx.send(TrainingEvent::Params(params.to_vec())).await;
                            },
                            Err(e) => {
                                error!("worker {id}: failed to send pull params command: {e}");

                                let event = TrainingEvent::Error(OrchErr::WorkerError {
                                    id: id,
                                    details: format!("failed to send pull params command: {e}"),
                                });

                                let _ = tx.send(event).await;
                                return;
                            },
                        }
                    }
                    Some(WorkerRequest::Disconnect) => {
                        if let Err(e) = worker_handle.disconnect().await {
                            error!("worker {id}: failed to send disconnect command: {e}");

                            let event = TrainingEvent::Error(OrchErr::WorkerError {
                                id: id,
                                details: format!("failed to send disconnect command: {e}"),
                            });

                            let _ = tx.send(event).await;
                        }
                        return;
                    }
                    None => {}
                },
                event = worker_handle.recv_event() => match event {
                    Ok(WorkerEvent::Loss(losses)) => {
                        debug!("worker {id} reported {} losses", losses.len());

                        let event = TrainingEvent::PublishedLosses {
                            worker_id: id,
                            losses,
                        };

                        let _ = tx.send(event).await;
                    }
                    Ok(WorkerEvent::Done) => {
                        info!("worker {id} done");
                        let _ = tx.send(TrainingEvent::WorkerDone(id)).await;
                    }
                    Ok(WorkerEvent::Disconnect) => {
                        info!("worker {id} disconnected");
                        return;
                    }
                    Ok(event) => {
                        warn!("worker {id}: unexpected event {event:?}");

                        let event = TrainingEvent::Error(OrchErr::WorkerError {
                            id: id,
                            details: format!("unexpected event {event:?}"),
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
                            id: id,
                            details: e.to_string(),
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
    pub fn event_listener(self, mut cancel_rx: Receiver<()>) -> Receiver<TrainingEvent> {
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
                    AlgorithmConfig::ParameterServer { .. } => {
                        Self::finalize_parameter_server(self.servers, &event_tx).await
                    }
                    AlgorithmConfig::AllReduce { .. } => {
                        Self::finalize_all_reduce(&mut wk_txs, &mut internal_rx).await
                    }
                };

                info!("received {} total parameters", params.len());

                let event = TrainingEvent::TrainingComplete {
                    model: TrainedModel {
                        params,
                        model,
                        input_size,
                    },
                    reason,
                };

                let _ = event_tx.send(event).await;
            });
        });

        event_rx
    }

    fn spawn_worker_listeners(
        workers: Vec<WorkerHandle<NetRtp>>,
        internal_tx: Sender<TrainingEvent>,
    ) -> Vec<Sender<WorkerRequest>> {
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
        cancel_rx: &mut Receiver<()>,
        internal_rx: &mut Receiver<TrainingEvent>,
        wk_txs: &mut [Sender<WorkerRequest>],
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

                    for tx in wk_txs.iter_mut() {
                        let _ = tx.send(WorkerRequest::Stop).await;
                    }
                }
                evt = internal_rx.recv() => {
                    let Some(event) = evt else {
                        break;
                    };

                    match event {
                        TrainingEvent::WorkerDone(id) => {
                            workers_done += 1;
                            let _ = event_tx.send(TrainingEvent::WorkerDone(id)).await;
                            if workers_done == n_workers {
                                break;
                            }
                        }
                        TrainingEvent::PublishedLosses { worker_id, losses } => {
                            if stop_reason.is_none() {
                                if let Some(cfg) = early_stopping {
                                    if let Some(sig) = tracker.record(worker_id, &losses) {
                                        if (sig.prev - sig.curr).abs() < *cfg.tolerance as f64 {
                                            info!(
                                                "early stopping triggered (prev={:.6}, curr={:.6})",
                                                sig.prev, sig.curr
                                            );

                                            stop_reason = Some(StopReason::EarlyStopping);
                                            for tx in wk_txs.iter_mut() {
                                                let _ = tx.send(WorkerRequest::Stop).await;
                                            }
                                        }
                                    }
                                }
                            }

                            let _ = event_tx.send(TrainingEvent::PublishedLosses { worker_id, losses }).await;
                        }
                        TrainingEvent::Error(e) => {
                            let _ = event_tx.send(TrainingEvent::Error(e)).await;
                            return None;
                        }
                        other => {
                            let _ = event_tx.send(other).await;
                        }
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

    async fn finalize_all_reduce(
        wk_txs: &mut Vec<Sender<WorkerRequest>>,
        internal_rx: &mut Receiver<TrainingEvent>,
    ) -> Vec<f32> {
        let mut i = 0;

        let params = loop {
            let _ = wk_txs[i].send(WorkerRequest::PullParams).await;

            if let Some(TrainingEvent::Params(params)) = internal_rx.recv().await {
                break params.to_vec();
            }

            i = (i + 1) % wk_txs.len();
        };

        for tx in wk_txs {
            let _ = tx.send(WorkerRequest::Disconnect).await;
        }

        params
    }

    /// Connects to the server nodes and sends their specification.
    ///
    /// # Args
    /// * `servers` - The servers' network addresses and specifications.
    /// * `connector` - The connector to create the network connections.
    ///
    /// # Returns
    /// The worker handles or an orch error if occurred.
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

    /// Connects to the worker nodes and sends their specification and dataset partitions.
    ///
    /// # Args
    /// * `workers` - The workers' network addresses, specifications and dataset partitions.
    /// * `partitions` - The workers' dataset partitions.
    /// * `connector` - The connector to create the network connections.
    ///
    /// # Returns
    /// The worker handles or an orch error if occurred.
    async fn create_workers<'a, F>(
        workers: Vec<(String, WorkerSpec, Partition<'a>)>,
        connector: &Connector<OwnedReadHalf, OwnedWriteHalf, NetRtp, F>,
    ) -> Result<Vec<WorkerHandle<NetRtp>>>
    where
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> NetRtp,
    {
        const CHUNK_SIZE: usize = 8192;

        let futs = workers
            .into_iter()
            .enumerate()
            .map(|(i, (addr, spec, partition))| async move {
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
                Self::send_partition(&mut worker_handle, partition, CHUNK_SIZE).await?;
                Ok::<_, OrchErr>(worker_handle)
            });

        future::try_join_all(futs).await
    }

    /// Sends a dataset partition to a worker.
    ///
    /// # Args
    /// * `worker_handle` - The handle for communicating with the worker.
    /// * `partition` - The dataset partition to send.
    /// * `chunk_size` - The chunk size to use for each message in bytes.
    ///
    /// # Returns
    /// An orch error if occurred.
    async fn send_partition<'a, T: TransportLayer>(
        worker_handle: &mut WorkerHandle<T>,
        partition: Partition<'a>,
        chunk_size: usize,
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
                    worker_handle,
                    &samples_path,
                    &labels_path,
                    samples_offset,
                    labels_offset,
                    samples_size,
                    labels_size,
                    chunk_size,
                )
                .await
            }
            Partition::Inline { samples, labels } => {
                Self::send_inline_partition(worker_handle, samples, labels, chunk_size).await
            }
        }
    }

    /// Sends a local partition to a worker.
    ///
    /// # Args
    /// * `worker_handle` - The handle for communicating with the worker.
    /// * `samples_path` - The path to the samples file.
    /// * `labels_path` - The path to the labels file.
    /// * `samples_offset` - The file offset to the starting position for this worker's samples.
    /// * `labels_offset` - The file offset to the starting position for this worker's labels.
    /// * `samples_size` - The amount of bytes this worker's samples take.
    /// * `labels_size` - The amount of bytes this worker's labels take.
    /// * `chunk_size` - The chunk size to use for each message in bytes.
    ///
    /// # Returns
    /// An orch error if occurred.
    async fn send_local_partition<T>(
        worker_handle: &mut WorkerHandle<T>,
        samples_path: &Path,
        labels_path: &Path,
        samples_offset: u64,
        labels_offset: u64,
        samples_size: u64,
        labels_size: u64,
        chunk_size: usize,
    ) -> Result<()>
    where
        T: TransportLayer,
    {
        let mut samples_fd = File::open(samples_path).await?;
        let mut labels_fd = File::open(labels_path).await?;

        samples_fd.seek(SeekFrom::Start(samples_offset)).await?;
        labels_fd.seek(SeekFrom::Start(labels_offset)).await?;

        worker_handle
            .push_dataset(
                &mut samples_fd.take(samples_size),
                &mut labels_fd.take(labels_size),
                samples_size as usize,
                labels_size as usize,
                chunk_size,
            )
            .await?;

        Ok(())
    }

    /// Sends in inline partition to a worker.
    ///
    /// # Args
    /// * `worker_handle` - The handle for communicating with the worker.
    /// * `samples` - The slice of samples.
    /// * `labels` - The slice of labels.
    /// * `chunk_size` - The chunk size to use for each message in bytes.
    ///
    /// # Returns
    /// An orch error if occurred.
    async fn send_inline_partition<T>(
        worker_handle: &mut WorkerHandle<T>,
        samples: &[f32],
        labels: &[f32],
        chunk_size: usize,
    ) -> Result<()>
    where
        T: TransportLayer,
    {
        let mut samples_cursor = share_dataset::get_dataset_cursor(samples);
        let mut labels_cursor = share_dataset::get_dataset_cursor(labels);

        worker_handle
            .push_dataset(
                &mut samples_cursor,
                &mut labels_cursor,
                samples.len(),
                labels.len(),
                chunk_size,
            )
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
