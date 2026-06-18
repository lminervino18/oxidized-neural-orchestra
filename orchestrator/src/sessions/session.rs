use std::{
    collections::{HashMap, HashSet},
    io::SeekFrom,
    path::Path,
    thread,
};

use comms::{
    Connector, NetRtp, ParamServerHandle, TransportLayer, WorkerHandle, protocol::Entity,
    share_dataset,
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
use uuid::Uuid;

use super::{EventListener, TrainedModel, WorkerListener, WorkerRequest};
use crate::{
    OrchErr, Result, StopReason, TrainingEvent,
    configs::{
        AlgorithmConfig, OrchAdapt, Partition, ServerAdapt, StrategySwitchTracking, WorkerAdapt,
    },
    sessions::{ConvergenceTracker, LossRecorder},
};

/// An ongoing training session.
pub struct Session {
    runtime: Runtime,
    orch_adapt: OrchAdapt,
    worker_handles: Vec<WorkerHandle<NetRtp>>,
    server_handles: Vec<ParamServerHandle<NetRtp>>,
}

impl Session {
    /// Creates a new session by connecting to all workers and servers.
    ///
    /// # Args
    /// * `orch` - The orchestrator's values for orchestrating the session.
    /// * `workers` - The workers' network addresses, specifications and dataset partitions.
    /// * `servers` - The servers' network addresses and specifications.
    /// * `connector` - The network connector.
    ///
    /// # Returns
    /// A ready session with all connections established.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or bootstrap message fails.
    pub fn new<F>(
        orch: OrchAdapt,
        workers: Vec<WorkerAdapt<'_>>,
        servers: Vec<ServerAdapt>,
        connector: Connector<OwnedReadHalf, OwnedWriteHalf, NetRtp, F>,
    ) -> Result<Self>
    where
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> NetRtp,
    {
        let runtime = Builder::new_multi_thread().enable_all().build()?;
        let (nworkers, nservers) = (workers.len(), servers.len());

        let server_handles = match orch.algorithm_config {
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

        let session = Self {
            runtime,
            orch_adapt: orch,
            worker_handles,
            server_handles,
        };

        Ok(session)
    }

    /// Consumes `self` and creates an event listener for this training session.
    ///
    /// Spawns a background task that drives the session. The `cancel_rx` must come
    /// from a `CancelHandle::pair()` call; the caller retains the `CancelHandle` to
    /// request an orderly stop at any time.
    ///
    /// # Args
    /// * `cancel_rx` - A training cancel event receiver.
    ///
    /// # Returns
    /// A receiver that yields `TrainingEvent`s as training progresses.
    pub fn event_listener(self, cancel_rx: Receiver<()>) -> Receiver<TrainingEvent> {
        let (user_event_tx, user_event_rx) = mpsc::channel(256);

        let Self {
            runtime,
            worker_handles,
            mut server_handles,
            orch_adapt:
                OrchAdapt {
                    input_size,
                    loss_recorder,
                    convergence_tracker,
                    model_config,
                    algorithm_config,
                    switch_tracking,
                    layer_param_offsets,
                },
        } = self;

        let run_loop_fut = async move {
            let (event_tx, mut event_rx) = mpsc::channel(256);

            let (Some(stop_reason), mut req_txs) = Self::start_training(
                worker_handles,
                &mut event_rx,
                &event_tx,
                cancel_rx,
                loss_recorder,
                convergence_tracker,
                &user_event_tx,
                switch_tracking,
                &mut server_handles,
            )
            .await
            else {
                return;
            };

            let params = Self::finalize_training(
                algorithm_config,
                server_handles,
                &user_event_tx,
                &mut req_txs,
                &mut event_rx,
                &layer_param_offsets,
            )
            .await;

            let nparams = params.len();
            info!("received {nparams} total parameters");

            let model = TrainedModel {
                params,
                model: model_config,
                input_size: input_size.get(),
            };

            let event = TrainingEvent::TrainingComplete { model, stop_reason };
            let _ = user_event_tx.send(event).await;
        };

        thread::spawn(move || runtime.block_on(run_loop_fut));
        user_event_rx
    }

    /// Starts the training stage for the orchestrator. Spawns the worker
    /// listeners and orchestrates the model's training.
    ///
    /// # Args
    /// * `worker_handles` - The handles for communicating with the workers.
    /// * `event_rx` - The worker listener event receiver.
    /// * `event_tx` - The event producer for the worker listeners.
    /// * `cancel_rx` - The user's halt event receiver.
    /// * `loss_recorder` - The workers' loss recorder.
    /// * `convergence_tracker` - A tracker device to track model convergence.
    /// * `user_event_tx` - The user event producer.
    /// * `switch_tracking` - The strategy switch tracking metadata.
    /// * `server_handles` - The server handles session vec.
    ///
    /// # Returns
    /// The worker listener requesters and the stopping reason for the training.
    async fn start_training(
        worker_handles: Vec<WorkerHandle<NetRtp>>,
        event_rx: &mut Receiver<TrainingEvent>,
        event_tx: &Sender<TrainingEvent>,
        cancel_rx: Receiver<()>,
        loss_recorder: LossRecorder,
        convergence_tracker: Option<ConvergenceTracker>,
        user_event_tx: &Sender<TrainingEvent>,
        switch_tracking: Option<StrategySwitchTracking>,
        server_handles: &mut Vec<ParamServerHandle<NetRtp>>,
    ) -> (Option<StopReason>, Vec<Sender<WorkerRequest>>) {
        let mut req_txs = Self::spawn_worker_listeners(worker_handles, event_tx);

        let mut event_listener = EventListener::new(
            cancel_rx,
            &mut req_txs,
            server_handles,
            loss_recorder,
            convergence_tracker,
            event_rx,
            user_event_tx.clone(),
            switch_tracking,
        );

        (event_listener.listen().await, req_txs)
    }

    /// Retrieves the parameters from the desired entity.
    ///
    /// # Args
    /// * `algorithm` - The current training algorithm.
    /// * `server_handles` - The handles for communicating with the servers.
    /// * `user_event_tx` - The user event notifier.
    /// * `req_txs` - The request senders for the worker listeners.
    /// * `event_rx` - The worker listener event consumer.
    /// * `layer_offsets` - Per-layer parameter locations: (server_id, start, end) within each server's buffer.
    ///
    /// # Returns
    /// The trained parameters of the model.
    async fn finalize_training<T>(
        algorithm: AlgorithmConfig,
        server_handles: Vec<ParamServerHandle<T>>,
        user_event_tx: &Sender<TrainingEvent>,
        req_txs: &mut [Sender<WorkerRequest>],
        event_rx: &mut Receiver<TrainingEvent>,
        layer_offsets: &[(Uuid, usize, usize)],
    ) -> Vec<f32>
    where
        T: TransportLayer + 'static,
    {
        match algorithm {
            AlgorithmConfig::ParameterServer { .. } => {
                Self::finalize_parameter_server(server_handles, layer_offsets, user_event_tx).await
            }
            AlgorithmConfig::AllReduce => {
                Self::finalize_all_reduce(req_txs, event_rx, user_event_tx).await
            }
            AlgorithmConfig::StrategySwitch { .. } if server_handles.is_empty() => {
                Self::finalize_all_reduce(req_txs, event_rx, user_event_tx).await
            }
            AlgorithmConfig::StrategySwitch { .. } => {
                Self::finalize_parameter_server(server_handles, layer_offsets, user_event_tx).await
            }
        }
    }

    /// Spawns the worker listeners.
    ///
    /// # Args
    /// * `worker_handles` - The handles for communicating with the workers.
    /// * `event_tx` - The listener's response sender.
    ///
    /// # Returns
    /// A list of senders to make requests to the listeners.
    fn spawn_worker_listeners(
        worker_handles: Vec<WorkerHandle<NetRtp>>,
        event_tx: &Sender<TrainingEvent>,
    ) -> Vec<Sender<WorkerRequest>> {
        let mut req_txs = Vec::with_capacity(worker_handles.len());

        for (i, worker_handle) in worker_handles.into_iter().enumerate() {
            let (req_tx, req_rx) = mpsc::channel(256);
            req_txs.push(req_tx);

            let worker_listener = WorkerListener::new(i, worker_handle);
            tokio::spawn(worker_listener.listen(req_rx, event_tx.clone()));
        }

        req_txs
    }

    /// Finalizes a training using parameter server. Retrieves the parameters from
    /// all the servers and reassembles them in the original layer order.
    ///
    /// Layers are distributed across servers by `balanced_partitions`, so pulling
    /// server-by-server and concatenating naively produces a shuffled parameter
    /// vector. This function collects each server's buffer first, then walks the
    /// `layer_offsets` table to copy each layer's slice into the output in the
    /// correct order.
    ///
    /// # Args
    /// * `server_handles` - The handles for communicating with the servers.
    /// * `layer_offsets` - Per-layer locations: `(server_id, start, end)` within
    ///   each server's parameter buffer, indexed by layer index.
    /// * `user_event_tx` - The user sender for communicating if an error occurred.
    ///
    /// # Returns
    /// The trained parameters of the model in layer order.
    async fn finalize_parameter_server<T>(
        server_handles: Vec<ParamServerHandle<T>>,
        layer_offsets: &[(Uuid, usize, usize)],
        user_event_tx: &Sender<TrainingEvent>,
    ) -> Vec<f32>
    where
        T: TransportLayer,
    {
        debug!("all workers done, reading final params from all servers");
        let mut server_params = HashMap::with_capacity(server_handles.len());

        let req_err = async |i, e| {
            let details = format!("unexpected error from server {i}: {e}");
            let err = OrchErr::ServerError(details);
            let event = TrainingEvent::Error(err);
            let _ = user_event_tx.send(event).await;
        };

        for (i, mut server_handle) in server_handles.into_iter().enumerate() {
            // TODO: Acá eventualmente hay que ver como manejamos las caídas de
            //       los servidores. Capaz conviene tener a mano al Acceptor y
            //       en caso de que un servidor no responda que se vuelva a
            //       levantar o algo.
            //
            //       Hay que ver cómo también resolvemos tema caídas simultaneas.
            //       Si se corta el internet por ejemplo por un rato después se
            //       van a intentar reconectar varios y necesitamos saber quién
            //       es quién.
            let server_id = server_handle.id();

            loop {
                if let Err(e) = server_handle.req_params().await {
                    req_err(i, e).await;
                    continue;
                }

                match server_handle.pull_params().await {
                    Ok(params) => {
                        debug!("server {i}: pulled {} params", params.len());
                        server_params.insert(server_id, params.to_vec());

                        if let Err(e) = server_handle.disconnect().await {
                            error!("Failed to disconnect server {i}: {e}");
                        }

                        break;
                    }
                    Err(e) => req_err(i, e).await,
                }
            }
        }

        let total: usize = layer_offsets
            .iter()
            .map(|&(_, start, end)| end - start)
            .sum();

        let mut model_params = Vec::with_capacity(total);

        for (layer_i, &(server_id, start, end)) in layer_offsets.iter().enumerate() {
            let params = &server_params[&server_id];
            debug!("layer {layer_i}: server {server_id} [{start}..{end}]");
            model_params.extend_from_slice(&params[start..end]);
        }

        model_params
    }

    /// Finalizes a training using all reduce. Retrieves the parameters from
    /// the first worker that sends them.
    ///
    /// # Args
    /// * `req_txs` - The request senders to ask for the parameters.
    /// * `event_rx` - The request receptor, to retrieve the parameters.
    /// * `user_event_tx` - The user sender for communicating if an error occurred.
    ///
    /// # Returns
    /// The trained parameters of the model.
    async fn finalize_all_reduce(
        req_txs: &mut [Sender<WorkerRequest>],
        event_rx: &mut Receiver<TrainingEvent>,
        user_event_tx: &Sender<TrainingEvent>,
    ) -> Vec<f32> {
        let n_workers = req_txs.len();
        let mut i = 0;

        let params = loop {
            let _ = req_txs[i].send(WorkerRequest::PullParams).await;

            match event_rx.recv().await {
                Some(TrainingEvent::Params(params)) => break params.to_vec(),
                Some(TrainingEvent::Error(e)) => {
                    let details = format!("failed to retrieve params from worker {i}: {e}");
                    let err = OrchErr::WorkerError { id: i, details };
                    let _ = user_event_tx.send(TrainingEvent::Error(err)).await;
                }
                _ => {}
            }

            if let Some(TrainingEvent::Params(params)) = event_rx.recv().await {
                break params.to_vec();
            }

            i = (i + 1) % n_workers;
        };

        for tx in req_txs.iter_mut() {
            let _ = tx.send(WorkerRequest::Disconnect).await;
        }

        let mut disconnected = HashSet::new();

        while disconnected.len() < n_workers {
            if let Some(TrainingEvent::Disconnect { worker_id }) = event_rx.recv().await {
                disconnected.insert(worker_id);
            }
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
    async fn create_servers<I, F>(
        servers: I,
        connector: &Connector<OwnedReadHalf, OwnedWriteHalf, NetRtp, F>,
    ) -> Result<Vec<ParamServerHandle<NetRtp>>>
    where
        I: IntoIterator<Item = ServerAdapt>,
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> NetRtp,
    {
        let mut handles = Vec::new();

        for ServerAdapt { addr, spec } in servers {
            debug!("connecting to server at {addr}");

            let stream =
                TcpStream::connect(&addr)
                    .await
                    .map_err(|e| OrchErr::ConnectionFailed {
                        addr: addr.clone(),
                        source: e,
                    })?;

            let (rx, tx) = stream.into_split();
            let node_handle = connector
                .connect_node(rx, tx, Entity::Orchestrator)
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
    async fn create_workers<'a, F, I>(
        workers: I,
        connector: &Connector<OwnedReadHalf, OwnedWriteHalf, NetRtp, F>,
    ) -> Result<Vec<WorkerHandle<NetRtp>>>
    where
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> NetRtp,
        I: IntoIterator<Item = WorkerAdapt<'a>>,
    {
        const CHUNK_SIZE: usize = 8192;

        let futs = workers.into_iter().map(|adapt| async move {
            let WorkerAdapt {
                addr,
                spec,
                partition,
            } = adapt;

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
                .connect_node(rx, tx, Entity::Orchestrator)
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
