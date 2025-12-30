use bytes::BufMut;
use std::io;

pub trait Serialize {
    fn serialize<B: BufMut>(&self, buf: &mut B) -> io::Result<()>;
}
