use anyhow::Result;

mod app;
mod state;
mod ui;

fn main() -> Result<()> {
    app::run::run()
}
