use std::io::{self, IoSlice};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Reads through the `reader` until filling the given buffer with data.
///
/// # Args
/// * `reader` - The reader from where to pull bytes.
/// * `out` - The buffer to write the data.
///
/// # Returns
/// The amount of bytes read or an io error if occurred.
pub async fn read_all<R: AsyncRead + Unpin>(reader: &mut R, out: &mut [u8]) -> io::Result<usize> {
    let mut read = 0;
    let chunk_size = out.len();

    while read < chunk_size {
        let n = reader.read(&mut out[read..]).await?;
        read += n;

        if n == 0 {
            break;
        }
    }

    Ok(read)
}

/// Writes all the given io slices by calling write vectored.
///
/// # Args
/// * `writer` - The writer where to push the bytes.
/// * `bufs` - The buffers to write.
///
/// # Returns
/// An io error if occurred.
pub async fn write_all_vectored<const N: usize, W: AsyncWrite + Unpin>(
    writer: &mut W,
    mut bufs: [IoSlice<'_>; N],
) -> io::Result<()> {
    let mut i = 0;

    while i < N {
        let mut written = writer.write_vectored(&bufs[i..]).await?;

        while i < N && written >= bufs[i].len() {
            written -= bufs[i].len();
            i += 1;
        }

        if written > 0 {
            bufs[i].advance(written);
        }
    }

    Ok(())
}
