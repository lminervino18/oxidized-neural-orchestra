use std::io::Write;

pub trait Serialize<W: Write> {
    fn serialize(&self, bytes: &mut W);
}
