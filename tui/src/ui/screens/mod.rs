pub mod config;
pub mod menu;
pub mod training;

use crossterm::event::KeyCode;
use ratatui::Frame;

use config::ConfigState;
use menu::MenuState;
use training::TrainingState;

/// An action produced by a screen in response to a key event.
pub enum Action {
    /// No state change required.
    None,
    /// The application should exit.
    Quit,
    /// The application should transition to a new screen.
    Transition(Screen),
}

/// The active screen of the TUI application.
pub enum Screen {
    /// The main menu.
    Menu(MenuState),
    /// The JSON configuration input flow.
    Config(ConfigState),
    /// The live training dashboard.
    Training(TrainingState),
}

impl Screen {
    /// Draws the active screen into the given frame.
    ///
    /// # Args
    /// * `f` - The ratatui frame to draw into.
    pub fn draw(&mut self, f: &mut Frame) {
        match self {
            Screen::Menu(s) => menu::draw(f, s),
            Screen::Config(s) => config::draw(f, s),
            Screen::Training(s) => training::draw(f, s),
        }
    }

    /// Dispatches a key event to the active screen.
    ///
    /// # Args
    /// * `key` - The key that was pressed.
    ///
    /// # Returns
    /// An `Action` indicating what the application should do next.
    pub fn handle_key(&mut self, key: KeyCode) -> Action {
        match self {
            Screen::Menu(s) => menu::handle_key(s, key),
            Screen::Config(s) => config::handle_key(s, key),
            Screen::Training(s) => training::handle_key(s, key),
        }
    }
}