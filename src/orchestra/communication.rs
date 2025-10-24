use crate::orchestra::error::SystemError;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct Communication {
    tcp_stream: Option<Arc<Mutex<TcpStream>>>,
}

impl Clone for Communication {
    fn clone(&self) -> Self {
        Communication {
            tcp_stream: self.tcp_stream.clone(),
        }
    }
}

impl Communication {
    /// Build from an already-connected TcpStream.
    /// Forces blocking mode and clears read timeout.
    pub fn new_from_stream(stream: TcpStream) -> Result<Self, SystemError> {
        stream
            .set_nonblocking(false)
            .map_err(|e| SystemError::ConnectionError(format!("Failed to set blocking: {}", e)))?;
        stream
            .set_read_timeout(None)
            .map_err(|e| SystemError::ConnectionError(format!("Failed to clear read timeout: {}", e)))?;

        Ok(Communication {
            tcp_stream: Some(Arc::new(Mutex::new(stream))),
        })
    }

    #[allow(dead_code)]
    /// Connect to address and return a blocking, no-timeout Communication.
    pub fn new(address: &str) -> Result<Self, SystemError> {
        let  stream = TcpStream::connect(address)
            .map_err(|e| SystemError::ConnectionError(format!("Failed to connect to TCP: {}", e)))?;

        stream
            .set_nonblocking(false)
            .map_err(|e| SystemError::ConnectionError(format!("Failed to set blocking: {}", e)))?;
        stream
            .set_read_timeout(None)
            .map_err(|e| SystemError::ConnectionError(format!("Failed to clear read timeout: {}", e)))?;

        Ok(Communication {
            tcp_stream: Some(Arc::new(Mutex::new(stream))),
        })
    }

#[allow(dead_code)]
    /// Switch blocking vs non-blocking (and clear read timeout if blocking).
    pub fn set_blocking(&self, blocking: bool) -> Result<(), SystemError> {
        if let Some(ref s) = self.tcp_stream {
            let stream = s
                .lock()
                .map_err(|_| SystemError::CommunicationError("Failed to lock TcpStream".into()))?;
            stream
                .set_nonblocking(!blocking)
                .map_err(|e| SystemError::ConnectionError(format!("set_blocking error: {}", e)))?;
            if blocking {
                stream
                    .set_read_timeout(None)
                    .map_err(|e| SystemError::ConnectionError(format!("clear timeout error: {}", e)))?;
            }
            Ok(())
        } else {
            Err(SystemError::CommunicationError(
                "No TCP stream available.".into(),
            ))
        }
    }

    /// Duplicate the underlying socket (independent file descriptor).
    pub fn try_clone(&self) -> Result<Self, SystemError> {
        let arc = self
            .tcp_stream
            .as_ref()
            .ok_or_else(|| SystemError::CommunicationError("No TCP stream".into()))?;

        let guard = arc
            .lock()
            .map_err(|_| SystemError::CommunicationError("Failed to lock TcpStream".into()))?;

        let cloned = guard
            .try_clone()
            .map_err(|e| SystemError::ConnectionError(format!("try_clone failed: {}", e)))?;

        cloned
            .set_nonblocking(false)
            .map_err(|e| SystemError::ConnectionError(format!("set blocking failed: {}", e)))?;
        cloned
            .set_read_timeout(None)
            .map_err(|e| SystemError::ConnectionError(format!("clear timeout failed: {}", e)))?;

        Ok(Communication {
            tcp_stream: Some(Arc::new(Mutex::new(cloned))),
        })
    }

    /// Create two Communications (read, write) from an accepted TcpStream.
    pub fn split_from_stream(stream: TcpStream) -> Result<(Self, Self), SystemError> {
        // new_from_stream configures blocking + no timeout
        let read_comm = Communication::new_from_stream(stream)?;
        let write_comm = read_comm.try_clone()?; // independent FD for writing
        Ok((read_comm, write_comm))
    }

    /// Create two Communications (read, write) by connecting to an address.
    pub fn split(address: &str) -> Result<(Self, Self), SystemError> {
        // Connect once and then duplicate the FD for the second handle
        let stream = TcpStream::connect(address)
            .map_err(|e| SystemError::ConnectionError(format!("Failed to connect to TCP: {}", e)))?;
        Communication::split_from_stream(stream)
    }

    /// Send bytes over TCP. Blocking write.
    pub fn send(&mut self, data: &[u8]) -> Result<(), SystemError> {
        if let Some(ref mut stream) = self.tcp_stream {
            let mut guard = stream
                .lock()
                .map_err(|_| SystemError::CommunicationError("Failed to lock TcpStream".into()))?;
            guard
                .write_all(data)
                .map_err(|e| SystemError::CommunicationError(format!("Failed to send data via TCP: {}", e)))?;
            guard
                .flush()
                .map_err(|e| SystemError::CommunicationError(format!("Failed to flush TCP stream: {}", e)))?;
            Ok(())
        } else {
            Err(SystemError::CommunicationError(
                "No TCP stream available to send data.".into(),
            ))
        }
    }

    /// Receive bytes from TCP. Blocking read (waits until some data arrives or EOF).
    pub fn receive(&mut self) -> Result<Vec<u8>, SystemError> {
        let mut buffer = vec![0; 512];
        if let Some(ref mut stream) = self.tcp_stream {
            let mut guard = stream
                .lock()
                .map_err(|_| SystemError::CommunicationError("Failed to lock TcpStream".into()))?;

            let bytes_read = guard
                .read(&mut buffer)
                .map_err(|e| SystemError::CommunicationError(format!("Failed to receive data via TCP: {}", e)))?;

            if bytes_read == 0 {
                return Err(SystemError::CommunicationError("Connection closed".into()));
            }

            buffer.truncate(bytes_read);
            Ok(buffer)
        } else {
            Err(SystemError::CommunicationError(
                "No TCP stream available to receive data.".into(),
            ))
        }
    }
}
