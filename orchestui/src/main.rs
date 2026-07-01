mod app;
mod config;
mod naming;
mod ui;

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::init();
    app::run::run()
}
