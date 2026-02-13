use std::io;

use crate::{
    Deserialize, Serialize,
    specs::{
        machine_learning::{DatasetSpec, dataset::ChunkSpec},
        server::ServerSpec,
        worker::WorkerSpec,
    },
};

use super::protocol::*;

/// The command for the `Control` variant of the `Msg` enum.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command<'a> {
    CreateServer(ServerSpec),
    #[serde(borrow)]
    CreateWorker(WorkerSpec<'a>),

    /// Reports a sequence of losses computed by the worker over its partial dataset
    /// after completing an epoch (i.e., one full pass over that partial dataset).
    ReportLoss {
        worker_id: usize,
        losses: Vec<f32>,
    },

    Disconnect,
}

/// The payload data for the `Data` variant of the `Msg` enum.
#[derive(Debug)]
pub enum Payload<'a> {
    Grad(&'a [f32]),
    Params(&'a mut [f32]),
}

/// The data chunk for the `Dataset` variant of the `Msg` enum.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Chunk<'a> {
    #[serde(borrow)]
    Header(DatasetSpec<'a>),
    Chunk(ChunkSpec<'a>),
}

/// The errors for the `Err` variant of the `Msg` enum.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    BufferSizeMismatch { expected: usize, got: usize },
    Fatal(String),
}

/// The application layer message for the entire system.
#[derive(Debug)]
pub enum Msg<'a> {
    Control(Command<'a>),
    Data(Payload<'a>),
    Dataset(Chunk<'a>),
    Err(Detail),
}

impl Msg<'_> {
    fn buf_is_too_small<T>(size: usize) -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("The given buffer is too small {size}, must at least be {HEADER_SIZE} bytes"),
        ))
    }

    fn invalid_header<T>(bytes: Header) -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Received invalid header kind bytes {bytes:?}"),
        ))
    }
}

impl<'a> Serialize<'a> for Msg<'a> {
    fn serialize(&'a self, buf: &mut Vec<u8>) -> Option<&'a [u8]> {
        match self {
            Msg::Err(detail) => {
                buf.extend_from_slice(&ERR);

                // SAFETY: Serialize impl for `Detail` is derived and not implemented
                //         by hand. Nor has a non string-key map inside.
                serde_json::to_writer(buf, &detail).unwrap();
                None
            }
            Msg::Control(cmd) => {
                buf.extend_from_slice(&CONTROL);

                // SAFETY: Serialize impl for `Command` is derived and not implemented
                //         by hand. Nor has a non string-key map inside.
                serde_json::to_writer(buf, &cmd).unwrap();
                None
            }
            Msg::Data(payload) => {
                let (header, nums): (_, &[_]) = match payload {
                    Payload::Grad(grad) => (&GRAD, grad),
                    Payload::Params(params) => (&PARAMS, params),
                };

                buf.extend_from_slice(header);
                Some(bytemuck::cast_slice(nums))
            }
            Msg::Dataset(chunk) => {
                match chunk {
                    Chunk::Header(spec) => {
                        buf.extend_from_slice(&DATASET_HEADER);

                        // SAFETY: Serialize impl for `DatasetSpec` is derived and not implemented
                        //         by hand. Nor has a non string-key map inside.
                        serde_json::to_writer(buf, &spec).unwrap();
                    }
                    Chunk::Chunk(spec) => {
                        buf.extend_from_slice(&CHUNK);

                        // SAFETY: Serialize impl for `ChunkSpec` is derived and not implemented
                        //         by hand. Nor has a non string-key map inside.
                        serde_json::to_writer(buf, &spec).unwrap();
                    }
                }

                None
            }
        }
    }
}

impl<'a> Deserialize<'a> for Msg<'a> {
    fn deserialize(buf: &'a mut [u8]) -> io::Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Self::buf_is_too_small(buf.len());
        }

        let (header, rest) = buf.split_at_mut(HEADER_SIZE);

        // SAFETY: We splitted the buffer to be of size `HEADER_SIZE` just above.
        let header: Header = header.try_into().unwrap();

        match header {
            ERR => {
                let detail = serde_json::from_slice(rest)?;
                Ok(Self::Err(detail))
            }
            CONTROL => {
                let cmd = serde_json::from_slice(rest)?;
                Ok(Self::Control(cmd))
            }
            GRAD | PARAMS => {
                let nums = bytemuck::cast_slice_mut(rest);

                let payload = match header {
                    GRAD => Payload::Grad(nums),
                    PARAMS => Payload::Params(nums),
                    _ => unreachable!(),
                };

                Ok(Self::Data(payload))
            }
            DATASET_HEADER | CHUNK => {
                let chunk = serde_json::from_slice(rest)?;
                Ok(Self::Dataset(chunk))
            }
            bytes => Self::invalid_header(bytes),
        }
    }
}
