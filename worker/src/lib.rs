//! Worker crate: networking + state + training loop.
//!
//! Goal: keep a single persistent weights buffer and interpret it via layouts/views.
//! Networking is isolated in `net`, orchestration in `loop_`.

pub mod config;
pub mod data;
pub mod loop_;
pub mod metrics;
pub mod model;
pub mod net;
pub mod schedule; 
pub mod state;
pub mod train;

pub use config::WorkerConfig;
pub use loop_::WorkerLoop;
pub use net::PsClient;
pub use state::WorkerState;
