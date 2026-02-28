/// The state necessary to make forward and backward passes through the network.
pub struct ServerParamsMetadata<'mw> {
    pub params: &'mw mut [f32],
    pub grad: &'mw mut [f32],
    pub acc_grad_buf: &'mw mut [f32],
}

impl<'mw> ServerParamsMetadata<'mw> {
    /// Creates a new `ServerParamsMetadata`.
    ///
    /// # Arguments
    /// * `params` - The mutable slice of this server's parameters.
    /// * `grad` - The server's dedicated gradient slice.
    /// * `acc_grad_buf` - The server's accumulated gradient buffer.
    ///
    /// # Returns
    /// A new `ServerParamsMetadata` instance.
    pub fn new(params: &'mw mut [f32], grad: &'mw mut [f32], acc_grad_buf: &'mw mut [f32]) -> Self {
        Self {
            params,
            grad,
            acc_grad_buf,
        }
    }
}
