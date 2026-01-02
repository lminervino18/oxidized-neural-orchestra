pub trait Serialize {
    /// Serializes self.
    ///
    /// If returns `Some` then it will use that returned
    /// slice instead of what's inside `buf`.
    ///
    /// # Arguments
    /// * `buf` - A writable vec of bytes.
    ///
    /// # Returns
    /// An optional slice of bytes.
    fn serialize<'a>(&'a self, buf: &mut Vec<u8>) -> Option<&'a [u8]>;
}
