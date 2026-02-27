mod app;
mod config;
mod state;
mod ui;

fn main() {
    if let Err(e) = app::run::run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}