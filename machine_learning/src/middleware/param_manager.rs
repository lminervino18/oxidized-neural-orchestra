use std::io;

use comms::{
    Deserialize,
    msg::{Msg, Payload},
};
use tokio::io::{AsyncRead, AsyncWrite};

use super::{Middleware, ServerMetadata};

/// The manager of parameters, this middleware's module manages the model's parameter retrieval from the
/// servers and selects which set of parameters to use for each layer of the model's trining when traversing
/// it's layers forwards and backwards.
pub struct ParamManager<'mw, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    middleware: &'mw mut Middleware<R, W>,
    cursors: Vec<usize>,
}

impl<'mw, R, W> ParamManager<'mw, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `ParamManager`.
    ///
    /// # Arguments
    /// * `middleware` - The middleware for the communication between the worker and the server.
    ///
    /// # Returns
    /// A new `ParamManager` instance.
    pub(super) fn new(middleware: &'mw mut Middleware<R, W>) -> Self {
        Self {
            cursors: vec![0; middleware.layer_sizes.len()],
            middleware,
        }
    }

    /// Creates a new `FrontIter` parameter iterator.
    ///
    /// The returned iterator iterates the model's layers forward.
    ///
    /// # Returns
    /// A new `FrontIter` instance.
    pub fn front(&mut self) -> FrontIter<'_, R, W> {
        let middleware = &mut self.middleware;
        self.cursors.fill(0);

        FrontIter {
            servers: &mut middleware.servers,
            server_ordering: &middleware.ordering,
            layer_sizes: &middleware.layer_sizes,
            cursors: &mut self.cursors,
            curr: 0,
        }
    }

    /// Creates a new `BackIter` parameter iterator.
    ///
    /// The returned iterator iterates the model's layers backwards.
    ///
    /// # Returns
    /// A new `Backiter` instance.
    pub fn back(&mut self) -> BackIter<'_, R, W> {
        let middleware = &mut self.middleware;
        self.cursors.fill(0);

        BackIter {
            servers: &mut middleware.servers,
            server_ordering: &middleware.ordering,
            layer_sizes: &middleware.layer_sizes,
            cursors: &mut self.cursors,
            curr: 0,
        }
    }
}

impl<R, W> Drop for ParamManager<'_, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Will invalidate all the incoming message buffers for the server.
    fn drop(&mut self) {
        for server in self.middleware.servers.iter_mut() {
            server.has_msg = false;
        }
    }
}

/// A model's layer iterator.
///
/// This iterator iterates the layers of a model from the front.
pub struct FrontIter<'pm, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    servers: &'pm mut [ServerMetadata<R, W>],
    server_ordering: &'pm [usize],
    layer_sizes: &'pm [usize],
    cursors: &'pm mut [usize],
    curr: usize,
}

impl<'pm, R, W> FrontIter<'pm, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Tries to yield the next layer's parameters and gradient.
    ///
    /// # Returns
    /// An option denoting if there still are more parameters and gradient of an io result.
    pub async fn next(&mut self) -> Option<io::Result<(&mut [f32], &mut [f32])>> {
        let server_id = *self.server_ordering.get(self.curr)?;
        let layer_size = self.layer_sizes[self.curr];
        let server = &mut self.servers[server_id];

        if !server.has_msg {
            if let Err(e) = server.rx.recv_into::<Msg<'_>, _>(&mut server.rx_buf).await {
                return Some(Err(e));
            }

            server.has_msg = true;
        }

        let params = match Msg::deserialize(&mut server.rx_buf) {
            Ok(Msg::Data(Payload::Params(params))) => params,
            Ok(_) => unimplemented!(),
            Err(e) => return Some(Err(e)),
        };

        let start = self.cursors[server_id];
        let end = start + layer_size;

        self.cursors[server_id] = end;
        self.curr += 1;

        Some(Ok((&mut params[start..end], &mut server.grad[start..end])))
    }
}

/// A model's layer iterator.
///
/// This iterator iterates the layers of a model from the back.
pub struct BackIter<'pm, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    servers: &'pm mut [ServerMetadata<R, W>],
    server_ordering: &'pm [usize],
    layer_sizes: &'pm [usize],
    cursors: &'pm mut [usize],
    curr: usize,
}

impl<'pm, R, W> BackIter<'pm, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Tries to yield the next layer's parameters and gradient.
    ///
    /// # Returns
    /// An option denoting if there still are more parameters and gradient of an io result.
    pub async fn next(&mut self) -> Option<io::Result<(&mut [f32], &mut [f32])>> {
        if self.curr == self.server_ordering.len() {
            return None;
        }

        let idx = self.server_ordering.len() - self.curr - 1;
        let server_id = self.server_ordering[idx];
        let layer_size = self.layer_sizes[idx];
        let server = &mut self.servers[server_id];

        if !server.has_msg {
            if let Err(e) = server.rx.recv_into::<Msg<'_>, _>(&mut server.rx_buf).await {
                return Some(Err(e));
            }

            server.has_msg = true;
        }

        let params = match Msg::deserialize(&mut server.rx_buf) {
            Ok(Msg::Data(Payload::Params(params))) => params,
            Ok(_) => unimplemented!(),
            Err(e) => return Some(Err(e)),
        };

        let start = self.cursors[server_id];
        let end = start + layer_size;

        self.cursors[server_id] = end;
        self.curr += 1;

        Some(Ok((&mut params[start..end], &mut server.grad[start..end])))
    }
}

#[cfg(test)]
mod tests {
    use futures::future;
    use tokio::io::{self, DuplexStream, ReadHalf, WriteHalf};

    use super::*;

    async fn mock_server(streams: Vec<DuplexStream>, sizes: Vec<usize>) {
        let futs = streams.into_iter().zip(sizes).map(async |(stream, size)| {
            let mut params = vec![0.0; size];
            let msg = Msg::Data(Payload::Params(&mut params));

            let (rx, tx) = io::split(stream);
            let (_, mut tx) = comms::channel(rx, tx);
            tx.send(&msg).await?;
            Ok::<_, io::Error>(())
        });

        future::try_join_all(futs).await.unwrap();
    }

    fn setup_test<'a>(
        nservers: usize,
        sizes: &[usize],
        layer_sizes: &[usize],
        ordering: &[usize],
    ) -> Middleware<ReadHalf<DuplexStream>, WriteHalf<DuplexStream>> {
        let (wk_streams, sv_streams): (Vec<_>, Vec<_>) =
            (0..nservers).map(|_| io::duplex(1024)).unzip();

        tokio::spawn(mock_server(sv_streams, sizes.to_vec()));
        let servers = wk_streams.into_iter().zip(sizes).map(|(stream, &size)| {
            let (rx, tx) = io::split(stream);
            let (rx, tx) = comms::channel(rx, tx);
            (rx, tx, size)
        });

        let layers = ordering.iter().cloned().zip(layer_sizes.iter().cloned());
        Middleware::new(servers, layers)
    }

    #[tokio::test]
    async fn front() {
        const SERVERS: usize = 2;
        const SIZES: [usize; SERVERS] = [19, 20];

        const LAYERS: usize = 4;
        const LAYER_SIZES: [usize; LAYERS] = [9, 15, 5, 10];
        const ORDERING: [usize; LAYERS] = [0, 1, 1, 0];

        let mut mw = setup_test(SERVERS, &SIZES, &LAYER_SIZES, &ORDERING);
        let mut pm = mw.param_manager();
        let mut front = pm.front();

        assert_eq!(front.next().await.unwrap().unwrap().0.len(), 9);
        assert_eq!(front.next().await.unwrap().unwrap().0.len(), 15);
        assert_eq!(front.next().await.unwrap().unwrap().0.len(), 5);
        assert_eq!(front.next().await.unwrap().unwrap().0.len(), 10);
    }

    #[tokio::test]
    async fn back() {
        const SERVERS: usize = 2;
        const SIZES: [usize; SERVERS] = [19, 20];

        const LAYERS: usize = 4;
        const LAYER_SIZES: [usize; LAYERS] = [9, 15, 5, 10];
        const ORDERING: [usize; LAYERS] = [0, 1, 1, 0];

        let mut mw = setup_test(SERVERS, &SIZES, &LAYER_SIZES, &ORDERING);
        let mut pm = mw.param_manager();
        let mut back = pm.back();

        assert_eq!(back.next().await.unwrap().unwrap().0.len(), 10);
        assert_eq!(back.next().await.unwrap().unwrap().0.len(), 5);
        assert_eq!(back.next().await.unwrap().unwrap().0.len(), 15);
        assert_eq!(back.next().await.unwrap().unwrap().0.len(), 9);
    }
}
