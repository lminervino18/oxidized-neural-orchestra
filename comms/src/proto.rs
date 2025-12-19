//! Implements a simple communication protocol, writes the length of the message first using 4 bytes.

use std::io::{self, Cursor};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::{Deserialize, Serialize};

/// Writes 4 bytes of length and the data into `sink` using `buf` for serialization.
pub async fn write_msg<T, W>(msg: &T, buf: &mut Vec<u8>, sink: &mut W) -> io::Result<()>
where
    T: Serialize,
    W: AsyncWrite + Unpin,
{
    buf.resize(4, 0);

    let start = buf.len();
    msg.serialize(buf)?;
    let end = buf.len();

    let len = (end - start) as u32;
    buf[..4].copy_from_slice(&len.to_be_bytes());
    sink.write_all(buf).await
}

/// Reads 4 bytes of length and then the data, returns the deserialized item using `buf` for deserialization.
pub async fn read_msg<T, R>(src: &mut R, buf: &mut Vec<u8>) -> io::Result<T>
where
    T: Deserialize,
    R: AsyncRead + Unpin,
{
    let mut size = [0; 4];
    src.read_exact(&mut size).await?;
    let len = u32::from_be_bytes(size) as usize;

    buf.resize(len, 0);
    src.read_exact(buf).await?;

    let mut cursor = Cursor::new(&buf);
    T::deserialize(&mut cursor)
}
