mod sink;
mod source;

pub use sink::Sink;
pub use source::Source;

/// The type behind the length prefix for the framed messages.
type LenType = u64;

/// The size of the `LenType` type.
const LEN_TYPE_SIZE: usize = size_of::<LenType>();
