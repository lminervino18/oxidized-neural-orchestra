use anyhow::Result;

mod app;
mod state;
mod ui;

fn main() -> Result<()> {
    // Placeholder: we will run the interactive terminal app next commit.
    let mut st = state::mock::MockState::new();
    st.tick();
    let _view = st.view();
    println!("tui state model ok");
    Ok(())
}
