use comms::{
    OnoReceiver, OnoSender,
    msg::{Msg, Payload},
};
use tokio::io::{self, AsyncRead, AsyncWrite};

pub struct Middleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    servers: Vec<ServerMetadata<R, W>>,
    server_ordering: Vec<usize>,
    layer_sizes: Vec<usize>,
}

pub struct ServerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    rx: OnoReceiver<R>,
    tx: OnoSender<W>,
    grad: Vec<f32>,
}

impl<'a, R, W> Middleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub fn new(
        servers: Vec<ServerMetadata<R, W>>,
        server_ordering: Vec<usize>,
        layer_sizes: Vec<usize>,
    ) -> Option<Self> {
        Some(Self {
            servers,
            server_ordering,
            layer_sizes,
        })
    }

    pub fn param_manager(&mut self) -> ParamManager<'_, R, W> {
        ParamManager {
            params: (0..self.servers.len()).map(|_| None).collect(),
            middleware: self,
        }
    }
}

pub struct ParamManager<'mw, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    middleware: &'mw mut Middleware<R, W>,
    params: Vec<Option<&'mw mut [f32]>>,
}

impl<'mw, R, W> ParamManager<'mw, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub fn front(&mut self) -> FrontIter<'mw, '_, R, W> {
        FrontIter {
            servers: &mut self.middleware.servers,
            server_ordering: &self.middleware.server_ordering,
            layer_sizes: &self.middleware.layer_sizes,
            params: &mut self.params,
            curr: 0,
        }
    }
}

pub struct FrontIter<'mw, 'pm, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    servers: &'pm mut [ServerMetadata<R, W>],
    server_ordering: &'pm [usize],
    layer_sizes: &'pm [usize],
    params: &'pm mut [Option<&'mw mut [f32]>],
    curr: usize,
}

impl<'mw, 'pm, R, W> FrontIter<'mw, 'pm, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub async fn next(&mut self) -> Option<io::Result<&mut [f32]>> {
        let server_id = *self.server_ordering.get(self.curr)?;
        self.curr += 1;

        // TODO (maybe): Tener un actor que escuche por todos los mensajes
        //               y multiplexe por distintos canales los distintos
        //               tipos de mensajes, aca abajo iria un recv del canal
        //               de gradientes nomas.
        if self.params[server_id].is_none() {
            let msg = match self.servers[server_id].rx.recv().await {
                Ok(msg) => msg,
                Err(e) => return Some(Err(e)),
            };

            match msg {
                Msg::Data(Payload::Params(params)) => {
                    self.params[server_id] = Some(params);
                }
                _ => unimplemented!(),
            }
        }

        let params = self.params[server_id].take()?;
        let mid = self.layer_sizes[server_id];

        let (head, tail) = params.split_at_mut(mid);
        self.params[server_id] = Some(tail);
        Some(Ok(head))
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{DuplexStream, ReadHalf, WriteHalf};

    use super::*;

    async fn mock_server(streams: Vec<DuplexStream>) {
        let chans: Vec<_> = streams
            .into_iter()
            .map(|stream| {
                let (rx, tx) = io::split(stream);
                let (rx, tx) = comms::channel(rx, tx);
                (rx, tx)
            })
            .collect();

        let fut = async |(mut rx, mut tx): (
            OnoReceiver<ReadHalf<DuplexStream>>,
            OnoSender<WriteHalf<DuplexStream>>,
        )|
               -> io::Result<()> {
            loop {
                match rx.recv().await? {
                    Msg::Data(Payload::Grad(grad)) => {
                        let msg = Msg::Data(Payload::Params(&mut grad.to_vec()));
                        tx.send(&msg).await?;
                    }
                    _ => todo!(),
                }
            }
        };

        let futs = chans.into_iter().map(fut);
        futures::future::try_join_all(futs).await.unwrap();
    }

    fn build_metadatas<'a>(
        wk_streams: Vec<DuplexStream>,
        sizes: &[usize],
    ) -> Vec<ServerMetadata<ReadHalf<DuplexStream>, WriteHalf<DuplexStream>>> {
        wk_streams
            .into_iter()
            .zip(sizes)
            .map(|(stream, &size)| {
                let (rx, tx) = io::split(stream);
                let (rx, tx) = comms::channel(rx, tx);
                ServerMetadata {
                    rx,
                    tx,
                    grad: vec![0.0; size],
                }
            })
            .collect()
    }

    fn setup_test<'a>(
        servers: usize,
        sizes: &[usize],
        layers: usize,
        layer_sizes: &[usize],
        ordering: &[usize],
    ) -> Middleware<ReadHalf<DuplexStream>, WriteHalf<DuplexStream>> {
        let (wk_streams, sv_streams): (Vec<_>, Vec<_>) =
            (0..servers).map(|_| io::duplex(1024)).unzip();

        tokio::spawn(mock_server(sv_streams));
        let metadatas = build_metadatas(wk_streams, sizes);
        Middleware::new(metadatas, ordering.to_vec(), layer_sizes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn test_name() {
        const SERVERS: usize = 2;
        const SIZES: [usize; SERVERS] = [10, 20];

        const LAYERS: usize = 4;
        const LAYER_SIZES: [usize; LAYERS] = [9, 15, 5, 10];
        const ORDERING: [usize; LAYERS] = [0, 1, 1, 0];

        let mut mw = setup_test(SERVERS, &SIZES, LAYERS, &LAYER_SIZES, &ORDERING);
        let mut pm = mw.param_manager();
        let mut front = pm.front();

        while let Some(params) = front.next().await {
            assert!(dbg!(params).is_ok());
        }
    }
}
