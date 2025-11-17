use std::{
    io,
    net::{Shutdown, TcpStream},
    sync::mpsc::{self, Receiver, SendError, Sender},
    thread::{self, JoinHandle},
};

use crate::{protocol, serialization::Serialize};

pub struct Sink<T> {
    id: usize,
    handle: Option<JoinHandle<io::Result<()>>>,
    sink: Sender<Option<T>>,
}

impl<T: Serialize<Vec<u8>> + Send + 'static> Sink<T> {
    /// Creates a new `Sink` instance.
    ///
    /// Will write all it's sent data into `writer`.
    ///
    /// The actual writes are performed in a separate thread.
    pub(super) fn new(id: usize, writer: TcpStream) -> Self {
        let (sink, source) = mpsc::channel();
        let handle = thread::spawn(move || Self::consume(id, source, writer));

        Self {
            id,
            sink,
            handle: Some(handle),
        }
    }

    /// Sends `data` through a channel into the writing thread.
    ///
    /// Returns `Some` if failed to do so.
    pub fn send(&mut self, data: T) -> Option<T> {
        self.sink
            .send(Some(data))
            .map_err(|SendError(x)| x.unwrap())
            .err()
    }

    /// Will consume from `source` and write it's incoming data onto `sink`.
    fn consume(id: usize, source: Receiver<Option<T>>, sink: TcpStream) -> io::Result<()> {
        let mut bytes = Vec::new();

        while let Ok(Some(x)) = source.recv() {
            bytes.clear();
            x.serialize(&mut bytes);

            if let Err(e) = protocol::write_bytes(&bytes, &sink) {
                eprintln!("failed to write bytes at sink {id}: {e}");
            }
        }

        let _ = sink.shutdown(Shutdown::Write);
        Ok(())
    }
}

impl<T> Drop for Sink<T> {
    fn drop(&mut self) {
        let _ = self.sink.send(None);

        if let Some(Err(e)) = self.handle.take().map(JoinHandle::join) {
            let id = self.id;
            eprintln!("failed to join consumption handle for sink {id}: {e:?}");
        }
    }
}
