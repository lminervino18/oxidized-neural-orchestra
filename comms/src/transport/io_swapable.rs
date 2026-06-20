use tokio::io::{AsyncRead, AsyncWrite};

pub trait IoSwapable<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Swaps `self`'s inner reader and writer.
    ///
    /// # Args
    /// * `reader` - The new reader.
    /// * `writer` - The new writer.
    fn swap(&mut self, reader: R, writer: W);
}
