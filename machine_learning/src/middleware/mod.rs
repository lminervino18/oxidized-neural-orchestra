mod param_manager;
mod server_metadata;

use comms::{OnoReceiver, OnoSender};
pub use param_manager::ParamManager;
pub(super) use server_metadata::ServerMetadata;

use tokio::io::{AsyncRead, AsyncWrite};

pub struct Middleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    servers: Vec<ServerMetadata<R, W>>,
    ordering: Vec<usize>,
    layer_sizes: Vec<usize>,
}

impl<'a, R, W> Middleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `Middleware`.
    ///
    /// # Arguments
    /// * `servers` - An iterable of tuples of (receiver, sender, server size).
    /// * `layer_sizes` - The sizes of the layers of the model and the id of the server they are retrieved from.
    ///
    /// # Returns
    /// A new `Middleware` instance.
    pub fn new<I, J>(servers: I, layers: J) -> Self
    where
        I: IntoIterator<Item = (OnoReceiver<R>, OnoSender<W>, usize)>,
        J: IntoIterator<Item = (usize, usize)>,
    {
        let metadatas = servers
            .into_iter()
            .map(|(rx, tx, size)| ServerMetadata::new(rx, tx, size))
            .collect();

        let (ordering, layer_sizes) = layers.into_iter().unzip();

        Self {
            servers: metadatas,
            ordering,
            layer_sizes,
        }
    }

    /// Creates a new `ParamManager`.
    ///
    /// # Returns
    /// A new `ParamManager` instance.
    pub fn param_manager(&mut self) -> ParamManager<'_, R, W> {
        ParamManager::new(self)
    }
}
