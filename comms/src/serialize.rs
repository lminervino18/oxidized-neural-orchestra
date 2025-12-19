use std::io::{self, Write};

pub trait Serialize {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()>;
}
