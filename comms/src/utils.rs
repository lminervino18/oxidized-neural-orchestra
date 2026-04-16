use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

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
