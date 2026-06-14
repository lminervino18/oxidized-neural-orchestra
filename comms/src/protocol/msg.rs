use std::{
    borrow::Cow,
    fmt::{self, Formatter},
    io,
};

use half::f16;
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{SeqAccess, Visitor},
};

use super::specs::{
    machine_learning::TrainerSpec,
    node::{NodeSpec, StatRequest, StatResponse},
    server::ServerSpec,
};

pub type Header = u32;
pub const HEADER_SIZE: usize = size_of::<Header>();

/// The payload data for the `Data` variant of the `Msg` enum.
#[derive(Debug)]
pub enum Payload<'a> {
    DenseGrad(&'a [f16]),
    SparseGrad(&'a [u8]),
    Params(&'a mut [f32]),
    Datachunk(&'a [f32]),
}

/// An enum of the different types of entities in the system.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Entity {
    Node { id: usize },
    Orchestrator,
    ParamServer { id: usize },
    Worker { id: usize },
}

/// The command for the `Control` variant of the `Msg` enum.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command<'a> {
    Connect(Entity),
    CreateNode {
        spec: NodeSpec,
    },
    Disconnect,
    Done,
    Eof,
    Ping,
    Pong,
    ReportLoss {
        #[serde(deserialize_with = "deserialize_null_as_nan")]
        losses: Cow<'a, [f64]>,
    },
    RequestParams,
    ShareDataset,
    ShareDatasetSize {
        size: usize,
    },
    StatsRequest {
        reqs: Vec<StatRequest>,
    },
    StatsResponse {
        stats: Vec<StatResponse>,
    },
    StopAfterEpoch,
    Switch {
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
        trainer_spec: TrainerSpec,
    },
    Upgrade {
        spec: ServerSpec,
        ranges: Vec<(usize, usize)>,
    },
    Upgraded,
}

/// The application layer message for the entire system.
#[derive(Debug)]
pub enum Msg<'a> {
    Control(Command<'a>),
    Data(Payload<'a>),
}

impl<'a> Msg<'a> {
    pub(super) fn buf_is_too_small<T>(size: usize) -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("The given buffer is too small {size}, must at least be {HEADER_SIZE} bytes"),
        ))
    }

    pub(super) fn invalid_kind_byte<T>(byte: u8) -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Received an invalid kind byte {byte}"),
        ))
    }

    /// Serializes the given message without any extra capabilities.
    ///
    /// # Args
    /// * `msg` - The message to serialize.
    /// * `out` - The output buffer.
    ///
    /// # Returns
    /// An optional zero copy slice.
    pub fn serialize(&'a self, out: &mut Vec<u8>) -> Option<&'a [u8]> {
        match self {
            Msg::Control(cmd) => {
                let header = (0 as Header).to_be_bytes();
                out.extend_from_slice(&header);

                // SAFETY: Serialize impl for `Command` is derived and not implemented
                //         by hand. Nor has a non string-key map inside.
                serde_json::to_writer(out, &cmd).unwrap();
                None
            }
            Msg::Data(payload) => {
                let (kind, data): (Header, &[_]) = match payload {
                    Payload::DenseGrad(grad) => (1, bytemuck::cast_slice(grad)),
                    Payload::SparseGrad(sparse) => (2, sparse),
                    Payload::Params(params) => (3, bytemuck::cast_slice(params)),
                    Payload::Datachunk(chunk) => (4, bytemuck::cast_slice(chunk)),
                };

                let header = kind.to_be_bytes();
                out.extend_from_slice(&header);
                Some(data)
            }
        }
    }

    /// Deserializes the given bytes and creates a new `Msg`.
    ///
    /// # Args
    /// * `data` - A serialized message.
    ///
    /// # Returns
    /// A result object that returns `Self` on success or `io::Error` on failure.
    pub fn deserialize(data: &'a mut [u8]) -> io::Result<Self> {
        if data.len() < HEADER_SIZE {
            return Msg::buf_is_too_small(data.len());
        }

        let (kind_buf, rest) = data.split_at_mut(HEADER_SIZE);

        // SAFETY: We splitted the buffer to be of size `HEADER_SIZE` just above.
        let kind = Header::from_be_bytes(kind_buf.try_into().unwrap()) as u8;

        match kind {
            0 => Ok(Msg::Control(serde_json::from_slice(rest)?)),
            1..5 => {
                let payload = match kind {
                    1 => Payload::DenseGrad(bytemuck::cast_slice(rest)),
                    2 => Payload::SparseGrad(rest),
                    3 => Payload::Params(bytemuck::cast_slice_mut(rest)),
                    4 => Payload::Datachunk(bytemuck::cast_slice(rest)),
                    _ => unreachable!(),
                };

                Ok(Msg::Data(payload))
            }
            byte => Msg::invalid_kind_byte(byte),
        }
    }
}

/// Deserializes the `ReportLoss` variant of the `Command` variant of messages.
///
/// # Args
/// * `deserializer` - The deserializar that serde will use to deserialize the loss report.
///
/// # Returns
/// Will always return the loss report, this deserialization can't fail.
fn deserialize_null_as_nan<'de, 'a, D>(deserializer: D) -> Result<Cow<'a, [f64]>, D::Error>
where
    D: Deserializer<'de>,
{
    struct LossVisitor;

    impl<'de> Visitor<'de> for LossVisitor {
        type Value = Vec<f64>;

        fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
            formatter.write_str("a sequence of float elements which may include nulls")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut values = seq.size_hint().map(Vec::with_capacity).unwrap_or_default();

            while let Some(loss) = seq.next_element::<Option<_>>()? {
                values.push(loss.unwrap_or(f64::NAN));
            }

            Ok(values)
        }
    }

    deserializer.deserialize_seq(LossVisitor).map(Cow::Owned)
}
