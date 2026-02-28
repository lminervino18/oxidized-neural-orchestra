/// The state necessary to make forward and backward passes through the network.
pub struct ServerParamsMetadata<'mw> {
    pub params: &'mw mut [f32],
    pub grad: &'mw mut [f32],
    pub acc_grad_buf: &'mw mut [f32],
}
