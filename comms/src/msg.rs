use std::{borrow::Cow, io};

use super::specs::{server::ServerSpec, worker::WorkerSpec};

pub type Header = u32;
pub const HEADER_SIZE: usize = size_of::<Header>();

/// The payload data for the `Data` variant of the `Msg` enum.
#[derive(Debug)]
pub enum Payload<'a> {
    Grad(&'a [f32]),
    Params(&'a mut [f32]),
    Datachunk(&'a [f32]),
}

/// The command for the `Control` variant of the `Msg` enum.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command<'a> {
    CreateServer(ServerSpec),
    CreateWorker(WorkerSpec),
    ReportLoss { losses: Cow<'a, [f32]> },
    StopAfterEpoch,
    Disconnect,
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
    Err(Detail),
}

impl Msg<'_> {
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
}
