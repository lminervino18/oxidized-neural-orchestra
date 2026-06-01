use std::ops::Deref;

/// A usize that must be greater than `1`.
#[derive(Debug, Clone, Copy)]
pub struct GreaterThanOneUsize {
    n: usize,
}

impl GreaterThanOneUsize {
    /// Creates a new `GreaterThanOneUsize`.
    ///
    /// # Args
    /// * `n` - The inner value.
    ///
    /// # Returns
    /// A new `GreaterThanOneUsize` instance if the given
    /// value is greater than `1`. `None` otherwise.
    pub fn new(n: usize) -> Option<Self> {
        (n > 1).then_some(Self { n })
    }
}

impl Deref for GreaterThanOneUsize {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.n
    }
}
