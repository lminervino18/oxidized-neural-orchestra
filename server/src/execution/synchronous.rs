use std::{borrow::Cow, sync::Arc};

use futures::future;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::Barrier,
};

use crate::{
    optimization::Optimizer,
    parameter_server::{PSClient, PSServer},
};
use comms::{
    OnoReceiver, OnoSender,
    msg::{Msg, Payload},
};

pub struct SyncExecutor<O>
where
    O: Optimizer,
{
    pserver: PSServer,
    optimizer: O,
}

impl<O> SyncExecutor<O>
where
    O: Optimizer + Send + 'static,
{
    pub fn new(pserver: PSServer, optimizer: O) -> Self {
        Self { pserver, optimizer }
    }

    pub async fn run<R, W, I>(mut self, conns: I)
    where
        R: AsyncRead + Unpin + 'static + Send,
        W: AsyncWrite + Unpin + 'static + Send,
        I: IntoIterator<Item = (OnoReceiver<R>, OnoSender<W>)>,
    {
        let (rxs, txs): (Vec<_>, Vec<_>) = conns.into_iter().unzip();

        let workers = rxs.len();
        let barrier = Arc::new(Barrier::new(1 + workers));

        let mut futs: Vec<_> = rxs
            .into_iter()
            .map(|rx| {
                let pclient = self.pserver.client_handle();
                let barrier = Arc::clone(&barrier);

                let pc_handle = Self::worker_handle(pclient, rx, barrier);
                tokio::spawn(pc_handle)
            })
            .collect();

        let ps_handle = self.server_handle(txs, barrier);
        futs.push(tokio::spawn(ps_handle));

        future::join_all(futs).await;
    }
}

impl<O> SyncExecutor<O>
where
    O: Optimizer,
{
    async fn server_handle<W>(mut self, mut txs: Vec<OnoSender<W>>, barrier: Arc<Barrier>)
    where
        W: AsyncWrite + Unpin,
    {
        loop {
            barrier.wait().await;

            self.pserver.update_weights(&mut self.optimizer);
            let weights = self.pserver.get_weights();
            let msg = Msg::Data(Payload::Weights(Cow::Borrowed(weights)));

            let gradient_push = txs.iter_mut().map(|tx| tx.send(&msg));
            future::join_all(gradient_push).await;
        }
    }

    async fn worker_handle<R>(pclient: PSClient, mut rx: OnoReceiver<R>, barrier: Arc<Barrier>)
    where
        R: AsyncRead + Unpin,
    {
        loop {
            let msg = match rx.recv().await {
                Ok(msg) => msg,
                Err(e) => {
                    // hubo un error de io
                    continue;
                }
            };

            match msg {
                Msg::Data(Payload::Gradient(grad, ..)) => pclient.accumulate(&grad).unwrap(),
                _ => unimplemented!(),
            }

            barrier.wait().await;
        }
    }
}
