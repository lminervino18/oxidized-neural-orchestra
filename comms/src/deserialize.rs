use std::io;

pub trait Deserialize<'a>: Sized {
    /// Deserializes the given bytes and creates a new Self.
    ///
    /// # Arguments
    /// * `buf` - A byte array.
    ///
    /// # Returns
    /// A result object that returns `Self` on success or `io::Error` on failure.
    fn deserialize(buf: &'a mut [u8]) -> io::Result<Self>;
}
