use std::io;

/// A trait for handles that are able to produce data sources.
pub trait DatasetSrc {
    /// Waits for the entity to send over a data source.
    ///
    /// # Args
    /// * `xs` - Where to write the incoming samples.
    /// * `ys` - Where to write the incoming labels.
    ///
    /// # Returns
    /// An io error if occurred.
    #[allow(async_fn_in_trait)]
    async fn pull_dataset(&mut self, xs: &mut Vec<f32>, ys: &mut Vec<f32>) -> io::Result<()>;
}
