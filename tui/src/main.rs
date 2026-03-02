mod app;
mod config;
mod ui;

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::init();
    app::run::run()
}