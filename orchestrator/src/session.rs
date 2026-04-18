use comms::{
    Connector, NetRtp, ParamServerHandle, PullParamsResponse, Rtp, WorkerEvent, WorkerHandle,
};
use futures::future;
use log::{debug, error, info, warn};
use std::path::PathBuf;
use std::{
    io::{self, Cursor},
    path::Path,
    slice, thread,
};
use tokio::io::AsyncRead;
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWrite},
    net::TcpStream,
    runtime::Runtime,
    sync::mpsc::{self, Receiver, Sender},
};

use comms::specs::{server::ServerSpec, worker::WorkerSpec};

use crate::{
    OrchErr, Result,
    configs::{LayerConfig, ModelConfig, Partition},
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
    servers: Vec<ParamServerHandle<NetRtp>>,
    workers: Vec<WorkerHandle<NetRtp>>,
    model: ModelConfig,
    input_size: usize,
}

impl Session {
    /// Creates a new session by connecting to all workers and servers.
    ///
    /// # Args
    /// * `workers` - List of (address, spec) pairs for each worker.
    /// * `partitions` - List of dataset partitions for each worker.
    /// * `servers` - List of (address, spec) pairs for each parameter server.
    /// * `connector` - The undrelying entity connector.
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
        connector: Connector,
        model: ModelConfig,
        input_size: usize,
    ) -> Result<Self> {
        let (nworkers, nservers) = (workers.len(), servers.len());
        info!("connecting to {nworkers} workers and {nservers} servers");

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        let server_chans = runtime.block_on(Self::create_servers(servers, connector))?;
        debug!("successfully created all servers");

        let worker_chans =
            runtime.block_on(Self::create_workers(workers, partitions, connector))?;
        debug!("successfully created all workers");

        Ok(Self {
            runtime,
            servers: server_chans,
            workers: worker_chans,
            model,
            input_size,
        })
    }

    async fn worker_listener(
        id: usize,
        mut worker_handle: WorkerHandle<NetRtp>,
        tx: Sender<TrainingEvent>,
    ) {
        loop {
            match worker_handle.recv_event().await {
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

        thread::spawn(move || {
            self.runtime.block_on(async move {
                let futs = self
                    .workers
                    .into_iter()
                    .enumerate()
                    .map(|(i, server_handle)| {
                        tokio::spawn(Self::worker_listener(i, server_handle, tx.clone()))
                    });

                future::join_all(futs).await;

                debug!("all workers done, reading final params from all servers");

                let mut model_params: Vec<f32> = Vec::new();

                for (i, mut server_handle) in self.servers.into_iter().enumerate() {
                    match server_handle.pull_params().await {
                        Ok(PullParamsResponse::Params(params)) => {
                            model_params.extend_from_slice(params);
                        }
                        Err(e) => {
                            let text = format!("unexpected error from server {i}: {e}");
                            let err = OrchErr::ServerError(text);
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
            });
        });

        rx
    }

    /// Connects to all parameter servers and sends each its bootstrap spec.
    ///
    /// # Args
    /// * `servers` - List of (address, spec) pairs for each parameter server.
    /// * `connector` - The connector for establishing connections.
    ///
    /// # Returns
    /// A list of open (receiver, sender) channel pairs, one per server.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or send fails.
    async fn create_servers(
        servers: Vec<(String, ServerSpec)>,
        connector: Connector,
    ) -> Result<Vec<ParamServerHandle<NetRtp>>> {
        let connect_to_server = async |id, addr| {
            let stream = TcpStream::connect(addr).await?;
            let (rx, tx) = stream.into_split();
            let server_handle = connector.connect_parameter_server(id, rx, tx).await?;
            Ok(server_handle)
        };

        let mut handles = Vec::with_capacity(servers.len());

        for (i, (addr, spec)) in servers.into_iter().enumerate() {
            let mut server_handle = connect_to_server(i, addr.clone())
                .await
                .map_err(|e| OrchErr::ConnectionFailed { addr, source: e })?;

            server_handle.create(spec).await?;
            handles.push(server_handle);
        }

        Ok(handles)
    }

    /// Connects to all workers, sends each its bootstrap spec and dataset partition.
    ///
    /// # Args
    /// * `workers` - List of (address, spec) pairs for each worker.
    /// * `partitions` - List of dataset partitions for each worker.
    /// * `connector` - The connector for establishing connections.
    ///
    /// # Returns
    /// A list of open (receiver, sender) channel pairs, one per worker.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or send fails.
    async fn create_workers<'a>(
        workers: Vec<(String, WorkerSpec)>,
        partitions: Vec<Partition<'a>>,
        connector: Connector,
    ) -> Result<Vec<WorkerHandle<NetRtp>>> {
        const CHUNK_SIZE: usize = 8192;

        let connect_to_worker = async |id, addr| {
            let stream = TcpStream::connect(addr).await?;
            let (rx, tx) = stream.into_split();
            let worker_handle = connector.connect_worker(id, rx, tx).await?;
            Ok(worker_handle)
        };

        let futs = workers.into_iter().zip(partitions).enumerate().map(
            |(i, ((addr, spec), partition))| async move {
                debug!("connecting to worker at {addr}");

                let mut worker_handle = connect_to_worker(i, addr.clone())
                    .await
                    .map_err(|e| OrchErr::ConnectionFailed { addr, source: e })?;

                worker_handle.create(spec).await?;

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
                            &mut worker_handle,
                        )
                        .await?;
                    }
                    Partition::Inline { samples, labels } => {
                        Self::send_inline_partition(
                            samples,
                            labels,
                            CHUNK_SIZE,
                            &mut worker_handle,
                        )
                        .await?;
                    }
                }

                Ok::<_, OrchErr>(worker_handle)
            },
        );

        let channels = future::try_join_all(futs).await?;
        Ok(channels)
    }

    async fn send_local_partition<R, W>(
        samples_path: &PathBuf,
        labels_path: &PathBuf,
        samples_offset: u64,
        labels_offset: u64,
        samples_size: u64,
        labels_size: u64,
        chunk_size: usize,
        worker_handle: &mut WorkerHandle<Rtp<R, W>>,
    ) -> Result<()>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
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

    async fn send_inline_partition<R, W>(
        samples: &[f32],
        labels: &[f32],
        chunk_size: usize,
        worker_handle: &mut WorkerHandle<Rtp<R, W>>,
    ) -> Result<()>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
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
