pub mod config;
pub mod worker;

pub use config::WorkerConfig;
pub use worker::Worker;

/// Initializes logging for the worker crate (env_logger + log).
/// Safe to call multiple times (subsequent calls will be ignored).
pub fn init_logging() {
    let _ = env_logger::try_init();
}
