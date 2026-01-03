//! Worker module: networking + state + training loop.
//!
//! Design goals:
//! - Single persistent `weights_buf` and `grads_buf` (no per-step allocations).
//! - Model is a "view" over buffers (implemented later in `model/*`).
//! - Networking details encapsulated in `net::client`.
//! - Loop orchestrates IO async + CPU-bound compute (spawn_blocking).

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
