/// The state necessary to make forward and backward passes through the network.
pub struct ServerParamsMetadata<'a> {
    pub params: &'a mut [f32],
    pub grad: &'a mut [f32],
}
