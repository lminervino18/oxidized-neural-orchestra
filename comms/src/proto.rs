//! Implements a simple communication protocol, writes the length of the message first.

use std::io::{self, Cursor};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::{Deserialize, Serialize};

type LenType = u64;
const LEN_SIZE: usize = size_of::<LenType>();

/// Writes bytes of length and the data into `sink` using `buf` for serialization.
pub async fn write_msg<T, W>(msg: &T, buf: &mut Vec<u8>, sink: &mut W) -> io::Result<()>
where
    T: Serialize,
    W: AsyncWrite + Unpin,
{
    buf.resize(LEN_SIZE, 0);

    let start = buf.len();
    msg.serialize(buf)?;
    let end = buf.len();

    let len = (end - start) as LenType;
    buf[..LEN_SIZE].copy_from_slice(&len.to_be_bytes());
    sink.write_all(buf).await
}

/// Reads bytes of length and then the data, returns the deserialized item using `buf` for deserialization.
pub async fn read_msg<T, R>(src: &mut R, buf: &mut Vec<u8>) -> io::Result<T>
where
    T: Deserialize,
    R: AsyncRead + Unpin,
{
    let mut size = [0; LEN_SIZE];
    src.read_exact(&mut size).await?;
    let len = LenType::from_be_bytes(size) as usize;

    buf.resize(len, 0);
    src.read_exact(buf).await?;

    let mut cursor = Cursor::new(&buf);
    T::deserialize(&mut cursor)
}
