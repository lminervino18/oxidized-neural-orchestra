use std::{io::SeekFrom, path::Path, thread};

use comms::{
    Connector, NetRtp, ParamServerHandle, TransportLayer, WorkerHandle,
    protocol::Entity,
    share_dataset,
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use futures::future;
use log::{debug, error, info};
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
    sessions::{EventListener, WorkerListener, WorkerRequest},
};

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
                info!("successfully created servers");
                server_handles
            }
            _ => Vec::new(),
        };

        info!("connecting to {nworkers} workers");
        let worker_handles = runtime.block_on(Self::create_workers(workers, &connector))?;
        info!("successfully created workers");

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

    /// Consumes `self` and creates an event listener for this training session.
    ///
    /// Spawns a background thread that drives the session. The `cancel_rx` must come
    /// from a `CancelHandle::pair()` call; the caller retains the `CancelHandle` to
    /// request an orderly stop at any time.
    ///
    /// # Returns
    /// A receiver that yields `TrainingEvent`s as training progresses.
    pub fn event_listener(self, cancel_rx: Receiver<()>) -> Receiver<TrainingEvent> {
        let (user_event_tx, user_event_rx) = mpsc::channel(256);

        let Self {
            workers,
            servers,
            model,
            algorithm,
            input_size,
            early_stopping,
            ..
        } = self;

        thread::spawn(move || {
            self.runtime.block_on(async move {
                let (req_tx, mut req_rx) = mpsc::channel(256);
                let mut req_txs = Vec::with_capacity(workers.len());

                for (i, worker_handle) in workers.into_iter().enumerate() {
                    let (tx, rx) = mpsc::channel(256);
                    req_txs.push(tx);

                    let worker_listener = WorkerListener::new(i, worker_handle);
                    tokio::spawn(worker_listener.listen(rx, req_tx.clone()));
                }

                let mut event_listener = EventListener::new(
                    cancel_rx,
                    &mut req_txs,
                    early_stopping,
                    &mut req_rx,
                    user_event_tx.clone(),
                );

                let Some(stop_reason) = event_listener.listen().await else {
                    return;
                };

                let params = match algorithm {
                    AlgorithmConfig::ParameterServer { .. } => {
                        Self::finalize_parameter_server(servers, user_event_tx.clone()).await
                    }
                    AlgorithmConfig::AllReduce { .. } => {
                        Self::finalize_all_reduce(&mut req_txs, &mut req_rx).await
                    }
                };

                info!("received {} total parameters", params.len());

                let event = TrainingEvent::TrainingComplete {
                    model: TrainedModel {
                        params,
                        model,
                        input_size,
                    },
                    reason: stop_reason,
                };

                let _ = user_event_tx.send(event).await;
            });
        });

        user_event_rx
    }

    async fn finalize_parameter_server(
        servers: Vec<ParamServerHandle<NetRtp>>,
        event_tx: Sender<TrainingEvent>,
    ) -> Vec<f32> {
        debug!("all workers done, reading final params from all servers");
        let mut model_params = Vec::new();

        for (i, mut server_handle) in servers.into_iter().enumerate() {
            match server_handle.pull_params().await {
                Ok(params) => {
                    model_params.extend_from_slice(params);

                    if let Err(e) = server_handle.disconnect().await {
                        error!("Failed to disconnect server {i}: {e}");
                    }
                }
                Err(e) => {
                    let details = format!("unexpected error from server {i}: {e}");
                    let err = OrchErr::ServerError(details);
                    let event = TrainingEvent::Error(err);
                    let _ = event_tx.send(event).await;
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
