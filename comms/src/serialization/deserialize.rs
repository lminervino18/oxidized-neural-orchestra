use std::fmt::{self, Display};

#[derive(Debug)]
pub struct DeError(String);

impl Display for DeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub type Result<T> = std::result::Result<T, DeError>;

pub trait Deserialize: Sized {
    fn deserialize(bytes: &[u8]) -> Result<Self>;
}
