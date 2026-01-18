pub trait Serialize<'a> {
    /// Serializes self.
    ///
    /// The `buf` vector is passed along so as to write data that cannot be zero-copied, if some serializable can skip
    /// being written to `buf` then this should be returned by this method.
    ///
    /// Then the `OnoSender` will first write the contents of `buf` and then whatever this method returns.
    ///
    /// # Arguments
    /// * `buf` - A writable vec of bytes.
    ///
    /// # Returns
    /// An optional slice of bytes.
    fn serialize(&'a self, buf: &mut Vec<u8>) -> Option<&'a [u8]>;
}
