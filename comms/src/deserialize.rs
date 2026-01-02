use std::io;

pub trait Deserialize<'a>: Sized {
    fn deserialize(buf: &'a [u8]) -> io::Result<Self>;
}
