//! The implementation of the sending end of the application layer protocol.

use std::io;

use tokio::io::AsyncWrite;

use crate::{Serialize, proto};

pub struct OnoSender<W>
where
    W: AsyncWrite + Unpin,
{
    tx: W,
    buf: Vec<u8>,
}

impl<W: AsyncWrite + Unpin> OnoSender<W> {
    /// Creates a new `OnoSender` instance.
    ///
    /// Will write all it's data through `tx`.
    pub fn new(tx: W) -> Self {
        Self {
            tx,
            buf: Vec::new(),
        }
    }

    /// Sends `msg` through the inner sender.
    pub async fn send<T: Serialize>(&mut self, msg: &T) -> io::Result<()> {
        proto::write_msg(msg, &mut self.buf, &mut self.tx).await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fmt::Debug,
        io::{Cursor, Write},
    };

    use super::*;
    use crate::Deserialize;

    impl Serialize for String {
        fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
            writer.write_all(self.as_bytes())
        }
    }

    async fn assert_message<T>(msg: T)
    where
        T: Debug + PartialEq + Serialize + Deserialize,
    {
        let mut sender = OnoSender::new(Vec::new());
        sender.send(&msg).await.unwrap();

        let got: T = proto::read_msg(&mut Cursor::new(sender.tx), &mut Vec::new())
            .await
            .unwrap();

        assert_eq!(msg, got)
    }

    #[tokio::test]
    async fn write_string() {
        assert_message("Hello World!".to_string()).await;
    }
}
