use std::io;

use crate::Align1;

pub trait Deserialize<'a>: Sized {
    /// Deserializes the given bytes and creates a new Self.
    ///
    /// # Args
    /// * `data` - A byte array.
    ///
    /// # Returns
    /// A result object that returns `Self` on success or `io::Error` on failure.
    fn deserialize<B: Align1>(data: &'a mut [B]) -> io::Result<Self>;
}
