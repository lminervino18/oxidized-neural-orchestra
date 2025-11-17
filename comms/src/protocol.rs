//! Implements a simple communication protocol, writes the length of the message first using 4 bytes.

use std::{
    io::{self, Read, Write},
    net::TcpStream,
};

/// Writes 4 bytes of length and then the data into the writer.
pub fn write_bytes(bytes: &[u8], mut writer: &TcpStream) -> io::Result<()> {
    let msg_len = bytes.len() as u32;
    writer.write_all(&msg_len.to_be_bytes())?;
    writer.write_all(bytes)?;
    writer.flush()?;
    Ok(())
}

/// Reads 4 bytes of length and then the data, then writes it into `bytes`.
pub fn read_bytes(bytes: &mut Vec<u8>, mut reader: &TcpStream) -> io::Result<()> {
    let mut msg_len_bytes = [0; size_of::<u32>()];
    reader.read_exact(&mut msg_len_bytes)?;

    let msg_len = u32::from_be_bytes(msg_len_bytes) as usize;
    bytes.resize(msg_len, 0);
    reader.read_exact(bytes)?;

    Ok(())
}
