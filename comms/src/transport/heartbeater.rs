use crate::{TransportLayer, protocol::Msg};
use std::{
    io,
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
pub struct KeepAliver<'a, R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    hearbeat_delay: Duration,
    messages_tx: Option<Sender<&'a Msg<'a>>>,
    messages_rx: Option<Receiver<Msg<'a>>>,
    handle: Option<JoinHandle<()>>,
    inner: Option<T>,
    _phantom: PhantomData<(R, W)>,
}

impl<R, W, T> KeepAliver<'_, R, W, T>
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
        let heartbeat_timeout = heartbeat_delay * 2; // ?
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
                    inner.send(&Msg::Heartbeat).await.unwrap(); // TODO
                }
                _ = sleep(heartbeat_timeout) => {
                    let now = Instant::now();

                    if last_seen + heartbeat_timeout < now {
                        break; // TODO: comunicar el timeout para arriba y además avisarle al peer
                               // por las dudas de q sea un falso positivo, como estamos en local no
                               // debería haber delay tal como para que pase esto pero por las dudas
                               // qcyo
                    }
                }
            }
        }
    }
}

// TODO: unwraps, no sé si convienen las options o qué
impl<R, W, T> TransportLayer<R, W> for KeepAliver<'_, R, W, T>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
    T: TransportLayer<R, W>,
{
    async fn recv(&mut self) -> io::Result<Msg<'_>> {
        match self.messages_rx.as_mut().unwrap().recv().await {
            Some(msg) => Ok(msg),
            None => {
                todo!() // esto podría ser el signaling de q el peer murió, también podría ser una
                // queue de results y listo no sé q queda mejor.
            }
        }
    }

    async fn send<'a>(&'a mut self, msg: &'a Msg<'_>) -> io::Result<()> {
        // acá no puedo mandar por la queue porq no puedo asegurar el lifetime
        // quizás esta tenga q ser la capa más baja o si no hay que splittear framer.
        // creo que esta capa también tiene que ser recon, porque la idea de mandar
        // heartbeats en async es justamente darse cuenta de q el peer cayó sin tener
        // que llamar send/recv. Ahora bien, si esta en una capa para arriba no tiene
        // acceso a comunicación con las task y habría q hacer alguna berretada como
        // intercomunicar las dos capas con una queue o algo así q es acoplarlas, o sea
        // debería ser la misma creo.
        // Hay un caso en el que se pierde un send mepa:
        // - se cae el peer
        // - hago send(msg)
        // - lo detecto
        // - me reconecto
        // - qué pasó con el send?
        // En general se supone que los send son mucho menos frecuentes que los heartbeats
        // como para que si se cae el peer pase esto
        // - se cae el peer
        // - lo detecto
        // - hago send(msg)
        // tipo no debería haber mucho tiempo muerto como para q el send caiga justo en el
        // medio, pero es un caso posbile, no sé bien cómo arreglarlo ahora
        // self.messages_tx.as_mut().unwrap().send(msg).await;
        Ok(())
    }

    fn swap(&mut self, reader: R, writer: W) {
        self.inner.as_mut().unwrap().swap(reader, writer);
    }

    fn demount(self) -> (R, W) {
        self.inner.unwrap().demount()
    }
}
