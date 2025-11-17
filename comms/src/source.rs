use std::{
    io,
    net::{Shutdown, TcpStream},
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
};

use crate::{protocol, serialization::Deserialize};

pub struct Source<T> {
    id: usize,
    handle: Option<JoinHandle<io::Result<()>>>,
    source: Receiver<T>,
    stream: TcpStream,
}

impl<T: Deserialize + Send + 'static> Source<T> {
    /// Creates a new `Source` instance.
    ///
    /// Will read all of it's data from the given `reader`.
    ///
    /// The actual reads are performed in a separate thread.
    pub(super) fn new(id: usize, reader: TcpStream) -> io::Result<Self> {
        let stream = reader.try_clone()?;
        let (sink, source) = mpsc::channel();
        let handle = thread::spawn(move || Self::consume(id, sink, reader));

        Ok(Self {
            id,
            source,
            handle: Some(handle),
            stream,
        })
    }

    /// Waits until any data is received from the inner channel.
    pub fn recv(&self) -> Option<T> {
        self.source.recv().ok()
    }

    /// Will read data from source, writing it into sink.
    fn consume(id: usize, sink: Sender<T>, source: TcpStream) -> io::Result<()> {
        let mut bytes = Vec::new();

        while protocol::read_bytes(&mut bytes, &source).is_ok() {
            match T::deserialize(&bytes) {
                Ok(x) => sink.send(x).unwrap(),
                Err(e) => eprintln!("failed to deserialize message at source {id}: {e}"),
            }

            bytes.clear();
        }

        Ok(())
    }
}

impl<T> Drop for Source<T> {
    fn drop(&mut self) {
        let _ = self.stream.shutdown(Shutdown::Read);

        if let Some(Err(e)) = self.handle.take().map(JoinHandle::join) {
            let id = self.id;
            eprintln!("failed to join consumption handle for source {id}: {e:?}");
        }
    }
}
