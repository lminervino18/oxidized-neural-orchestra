use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use crate::ui::screens::{menu, Screen};

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

/// Runs the TUI application.
///
/// # Errors
/// Returns an error if terminal setup or rendering fails.
pub fn run() -> Result<()> {
    let _guard = TerminalGuard::enter()?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut screen = Screen::Menu(menu::MenuState::new());

    loop {
        terminal.draw(|f| screen.draw(f))?;

        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match screen.handle_key(k.code) {
                    crate::ui::screens::Action::Quit => break,
                    crate::ui::screens::Action::Transition(next) => screen = next,
                    crate::ui::screens::Action::None => {}
                }
            }
        }
    }

    terminal.show_cursor()?;
    Ok(())
}
