use std::io;

use crate::Align1;

pub trait Deserialize<'a>: Sized {
    /// Deserializes the given bytes and creates a new Self.
    ///
    /// # Arguments
    /// * `buf` - A byte array.
    ///
    /// # Returns
    /// A result object that returns `Self` on success or `io::Error` on failure.
    fn deserialize<B: Align1>(buf: &'a mut [B]) -> io::Result<Self>;
}
