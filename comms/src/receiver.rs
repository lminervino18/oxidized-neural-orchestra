//! The implementation of the receiving end of the application layer protocol.

use std::io;

use tokio::io::AsyncRead;

use crate::{Deserialize, proto};

pub struct OnoReceiver<R: AsyncRead + Unpin> {
    rx: R,
    buf: Vec<u8>,
}

impl<R: AsyncRead + Unpin> OnoReceiver<R> {
    /// Creates a new `OnoReceiver` instance.
    ///
    /// Will read all it's data from `rx`.
    pub fn new(rx: R) -> Self {
        Self {
            rx,
            buf: Vec::new(),
        }
    }

    /// Waits to receive a new message from the inner receiver.
    pub async fn recv<T: Deserialize>(&mut self) -> io::Result<T> {
        proto::read_msg(&mut self.rx, &mut self.buf).await
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt::Debug, io::Read};

    use crate::serialize::Serialize;

    use super::*;

    impl Deserialize for String {
        fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
            let mut buf = String::new();
            reader.read_to_string(&mut buf)?;
            Ok(buf)
        }
    }

    async fn assert_message<T>(msg: T)
    where
        T: Debug + PartialEq + Serialize + Deserialize,
    {
        let mut payload = Vec::new();

        proto::write_msg(&msg, &mut Vec::new(), &mut payload)
            .await
            .unwrap();

        let mut receiver = OnoReceiver::new(payload.as_slice());
        let got: T = receiver.recv().await.unwrap();
        assert_eq!(msg, got);
    }

    #[tokio::test]
    async fn read_string() {
        assert_message("Hello World!".to_string()).await;
    }
}
