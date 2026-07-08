use std::{
    mem,
    sync::mpsc::Sender,
    time::{Duration, Instant},
};

use futures::io;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::Mutex,
    task::{self, JoinHandle},
    time::sleep,
};

use crate::{
    codec::{Sink, Source},
    protocol::Msg,
};

#[derive(Debug)]
struct Heartbeater<'a, R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    heartbeat_delay: Duration,
    heartbeat_timeout: Duration,
    handle: Option<JoinHandle<io::Result<()>>>,
    messages_tx: Option<Sender<Msg<'a>>>,
    messages_rx: Option<Source<R>>,
    heartbeats_tx: Option<Mutex<Sink<W>>>,
}

impl<'a, R, W> Heartbeater<'a, R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    pub fn new(
        heartbeat_delay: Duration,
        messages_tx: Sender<Msg<'a>>,
        messages_rx: Source<R>,
        heartbeats_tx: Mutex<Sink<W>>,
    ) -> Self {
        let heartbeat_timeout = heartbeat_delay * 2;
        let handle = None;

        Self {
            heartbeat_delay,
            heartbeat_timeout,
            handle,
            messages_tx: Some(messages_tx),
            messages_rx: Some(messages_rx),
            heartbeats_tx: Some(heartbeats_tx),
        }
    }

    pub fn start(&mut self) {
        let heartbeat_delay = self.heartbeat_delay;

        // SAFETY: Heartbeater is meant to be started just once.
        let messages_tx = self.messages_tx.take().unwrap();
        let messages_rx = self.messages_rx.take().unwrap();
        let heartbeats_tx = self.heartbeats_tx.take().unwrap();

        let handle = task::spawn(async move {
            Self::run(heartbeat_delay, messages_tx, messages_rx, heartbeats_tx).await
        });

        self.handle = Some(handle);
    }

    async fn run<'b>(
        heartbeat_delay: Duration,
        messages_tx: Sender<Msg<'b>>,
        mut messages_rx: Source<R>,
        heartbeats_tx: Mutex<Sink<W>>,
    ) -> io::Result<()> {
        let heartbeat_timeout = heartbeat_delay * 2; // ?
        let mut last_seen = Instant::now();

        loop {
            tokio::select! {
                msg = messages_rx.recv() => {
                    last_seen = Instant::now();

                    let Ok(msg) = msg else { todo!() };

                    // SAFETY: The message's inner lifetime outlives '1.
                    let msg = unsafe { mem::transmute::<Msg<'_>, Msg<'_>>(msg) };

                    if let Msg::Heartbeat = msg  {
                        continue
                    }

                    messages_tx.send(msg);
                },
                _ = sleep(heartbeat_delay) => {
                    let mut heartbeats_tx = heartbeats_tx.lock().await;
                    heartbeats_tx.send(&Msg::Heartbeat).await.unwrap(); // TODO
                }
                _ = sleep(heartbeat_timeout) => {
                    let now = Instant::now();

                    if last_seen + heartbeat_timeout < now {
                        // TODO: comunicar el timeout para arriba y además avisarle al peer
                        // por las dudas de q sea un falso positivo, como estamos en local no
                        // debería haber delay tal como para que pase esto pero por las dudas
                        // qcyo
                        let err_kind = io::ErrorKind::ConnectionAborted;
                        let detail = "Lost connection with peer";
                        return Err(io::Error::new(err_kind, detail))
                    }
                }
            }
        }
    }
}
