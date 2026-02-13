type HeaderType = u32;
pub const HEADER_SIZE: usize = size_of::<HeaderType>();
pub type Header = [u8; HEADER_SIZE];

const ERR_H: HeaderType = 0;
const CONTROL_H: HeaderType = 1;
const GRAD_H: HeaderType = 2;
const PARAMS_H: HeaderType = 3;
const DATASET_HEADER_H: HeaderType = 4;
const CHUNK_H: HeaderType = 5;

pub const ERR: Header = ERR_H.to_be_bytes();
pub const CONTROL: Header = CONTROL_H.to_be_bytes();
pub const GRAD: Header = GRAD_H.to_be_bytes();
pub const PARAMS: Header = PARAMS_H.to_be_bytes();
pub const DATASET_HEADER: Header = DATASET_HEADER_H.to_be_bytes();
pub const CHUNK: Header = CHUNK_H.to_be_bytes();
