use crate::{TransportLayer, protocol::Msg};
use std::{
    marker::PhantomData,
    mem,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::mpsc::{self, Receiver, Sender},
    task::{self, JoinHandle},
    time::sleep,
};

const BUFF_SIZE: usize = 1024; // TODO

#[derive(Debug)]
pub struct Hearbeater<'a, R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    hearbeat_delay: Duration,
    messages_tx: Option<Sender<Msg<'a>>>,
    messages_rx: Option<Receiver<Msg<'a>>>,
    handle: Option<JoinHandle<()>>,
    inner: Option<T>,
    _phantom: PhantomData<(R, W)>,
}

impl<R, W, T> Hearbeater<'_, R, W, T>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
    T: TransportLayer<R, W> + 'static,
{
    pub fn new(hearbeat_delay: Duration, inner: T) -> Self {
        Self {
            hearbeat_delay,
            messages_tx: None,
            messages_rx: None,
            handle: None,
            inner: Some(inner),
            _phantom: PhantomData,
        }
    }

    pub fn start(&mut self) {
        let (incoming_tx, incoming_rx) = mpsc::channel(BUFF_SIZE);
        let (outgoing_tx, outgoing_rx) = mpsc::channel(BUFF_SIZE);

        let inner = self.inner.take().unwrap();
        let heartbeat_delay = self.hearbeat_delay.clone();

        let handle = task::spawn(async move {
            Self::run(incoming_tx, outgoing_rx, heartbeat_delay, inner).await;
        });

        todo!();
        // self.handle = Some(handle);
        // self.messages_rx = Some(incoming_rx);
        // self.messages_rx = Some(incoming_rx);
    }

    async fn run(
        mut incoming_tx: Sender<Msg<'_>>,
        mut outgoing_rx: Receiver<Msg<'_>>,
        heartbeat_delay: Duration,
        mut inner: impl TransportLayer<R, W> + 'static,
    ) {
        let mut last_seen = Instant::now();

        loop {
            tokio::select! {
                msg = inner.recv() => {
                    last_seen = Instant::now();

                    let Ok(msg) = msg else { todo!() };

                    // SAFETY: The message's inner lifetime outlives '1.
                    let msg = unsafe { mem::transmute::<Msg<'_>, Msg<'_>>(msg) };

                    // if msg == heartbeat { ... }

                    // incoming_tx.send(msg);
                    todo!()
                },
                _ = sleep(heartbeat_delay) => {
                    peer_alive = false;
                    inner.send(&Msg::Heartbeat).await.unwrap(); // TODO
                }
            }
        }
    }
}
