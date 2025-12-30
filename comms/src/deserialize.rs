use bytes::Buf;
use std::io;

pub trait Deserialize<'a>: Sized {
    fn deserialize<B: Buf>(buf: &'a mut B) -> io::Result<Self>;
}
