use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::state::mock::MockState;
use crate::ui::draw::draw;

/// Runs the interactive TUI.
///
/// # Errors
/// Returns an error if terminal setup, event polling, or rendering fails.
pub fn run() -> Result<()> {
    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = MockState::new();
    let mut show_diagram = true;
    let mut show_logs = true;

    // Main loop
    loop {
        state.tick();
        let view = state.view();

        terminal.draw(|f| {
            draw(f, &view, show_diagram, show_logs);
        })?;

        // Input handling with timeout tick
        if event::poll(Duration::from_millis(120))? {
            match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => match k.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('d') => show_diagram = !show_diagram,
                    KeyCode::Char('l') => show_logs = !show_logs,
                    _ => {}
                },
                _ => {}
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
