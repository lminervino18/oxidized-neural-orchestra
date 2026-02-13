type HeaderType = u32;
pub const HEADER_SIZE: usize = size_of::<HeaderType>();
pub type Header = [u8; HEADER_SIZE];

const ERR_H: HeaderType = 0;
const CONTROL_H: HeaderType = 1;
const GRAD_H: HeaderType = 2;
const PARAMS_H: HeaderType = 3;

pub const ERR: Header = ERR_H.to_be_bytes();
pub const CONTROL: Header = CONTROL_H.to_be_bytes();
pub const GRAD: Header = GRAD_H.to_be_bytes();
pub const PARAMS: Header = PARAMS_H.to_be_bytes();
