use crate::orchestra::error::SystemError; // Importing the SystemError struct for error handling
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex}; // For thread-safety when sharing TcpStream

#[derive(Debug)]
/// Structure that handles TCP communication.
///
/// This structure wraps a `TcpStream` and provides methods to send
/// and receive data over a TCP connection. The connection is established
/// externally and passed into the structure when creating a new `Communication` instance.
///
pub struct Communication {
    tcp_stream: Option<Arc<Mutex<TcpStream>>>, // Wrap TcpStream in Arc<Mutex> for thread safety
}

impl Clone for Communication {
    fn clone(&self) -> Self {
        // Clone the Arc (not the TcpStream itself)
        Communication {
            tcp_stream: self.tcp_stream.clone(),
        }
    }
}

impl Communication {
    /// Creates a new `Communication` instance from an already established `TcpStream`.
    ///
    /// # Parameters
    /// - `stream`: A pre-established `TcpStream` to be used for communication.
    ///
    /// # Returns
    /// A `Result` containing a `Communication` instance on success, or a `SystemError` on failure.
    pub fn new_from_stream(stream: TcpStream) -> Result<Self, SystemError> {
        // Wrap the TcpStream in Arc<Mutex> for safe shared access
        Ok(Communication {
            tcp_stream: Some(Arc::new(Mutex::new(stream))),
        })
    }

    /// Creates a new `Communication` instance by connecting to a specified address.
    ///
    /// # Parameters
    /// - `address`: A string containing the address in the format "ip:port" (e.g., "127.0.0.1:5000").
    ///
    /// # Returns
    /// A `Result` containing a `Communication` instance on success, or a `SystemError` on failure.
    pub fn new(address: &str) -> Result<Self, SystemError> {
        // Attempt to connect to the specified address and establish a TCP stream
        let stream = TcpStream::connect(address).map_err(|e| {
            SystemError::ConnectionError(format!("Failed to connect to TCP: {}", e))
        })?;

        // Set a read timeout on the connection
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .map_err(|e| {
                SystemError::ConnectionError(format!("Failed to set read timeout: {}", e))
            })?;

        // Wrap the TcpStream in Arc<Mutex> for safe shared access
        Ok(Communication {
            tcp_stream: Some(Arc::new(Mutex::new(stream))),
        })
    }

    /// Sends data over the TCP connection.
    ///
    /// # Parameters
    /// - `data`: The data to send, as a byte slice (`&[u8]`).
    ///
    /// # Returns
    /// A `Result<(), SystemError>` indicating success or failure.
    pub fn send(&mut self, data: &[u8]) -> Result<(), SystemError> {
        // Check if the TCP stream is available
        if let Some(ref mut stream) = self.tcp_stream {
            // Send the data over the TCP stream
            println!("Sending data: {:?}", data);
            stream.lock().unwrap().write_all(data).map_err(|e| {
                SystemError::CommunicationError(format!("Failed to send data via TCP: {}", e))
            })?;
            stream.lock().unwrap().flush().map_err(|e| {
                SystemError::CommunicationError(format!("Failed to flush TCP stream: {}", e))
            })?;
        } else {
            return Err(SystemError::CommunicationError(
                "No TCP stream available to send data.".into(),
            ));
        }
        Ok(())
    }

    /// Receives data from the TCP connection.
    ///
    /// # Returns
    /// A `Result<Vec<u8>, SystemError>` containing received data or an error.
    pub fn receive(&mut self) -> Result<Vec<u8>, SystemError> {
        let mut buffer = vec![0; 512]; // Buffer to store the received data
        if let Some(ref mut stream) = self.tcp_stream {
            let bytes_read = stream.lock().unwrap().read(&mut buffer).map_err(|e| {
                SystemError::CommunicationError(format!("Failed to receive data via TCP: {}", e))
            })?;

            // If we read 0 bytes, it means the connection has been closed
            if bytes_read == 0 {
                println!("Connection closed by the other side.");
                return Err(SystemError::CommunicationError("Connection closed".into())); // Or handle it as needed
            }

            // Debug print to check the number of bytes read and the buffer content
            println!("Received {} bytes: {:?}", bytes_read, &buffer[..bytes_read]);

            buffer.truncate(bytes_read); // Adjust the buffer size to the number of bytes read
        } else {
            return Err(SystemError::CommunicationError(
                "No TCP stream available to receive data.".into(),
            ));
        }
        Ok(buffer)
    }
}
