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

pub trait Demountable<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Consumes `self` and yields the inner io reader and writer.
    ///
    /// # Returns
    /// Self's both reader and writer.
    fn demount(self) -> (R, W);
}
