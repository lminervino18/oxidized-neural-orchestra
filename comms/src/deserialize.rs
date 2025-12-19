use std::io::{self, Read};

pub trait Deserialize: Sized {
    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self>;
}
