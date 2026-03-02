use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use crate::ui::screens::{menu::MenuState, Action, Screen};

/// RAII guard that enables raw mode and the alternate screen on construction,
/// and restores the terminal to its original state on drop.
struct TerminalGuard;

impl TerminalGuard {
    /// Enters raw mode and switches to the alternate screen.
    ///
    /// # Errors
    /// Returns an error if either terminal operation fails.
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

    let mut screen = Screen::Menu(MenuState::new());

    loop {
        terminal.draw(|f| screen.draw(f))?;

        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match screen.handle_key(k.code) {
                    Action::Quit => break,
                    Action::Transition(next) => screen = next,
                    Action::None => {}
                }
            }
        }
    }

    terminal.show_cursor()?;
    Ok(())
}