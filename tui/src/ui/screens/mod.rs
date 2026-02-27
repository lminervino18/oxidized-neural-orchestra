pub mod config;
pub mod menu;
pub mod training;

use crossterm::event::KeyCode;
use ratatui::Frame;

pub enum Action {
    None,
    Quit,
    Transition(Screen),
}

pub enum Screen {
    Menu(menu::MenuState),
    Config(config::ConfigState),
    Training(training::TrainingState),
}

impl Screen {
    pub fn draw(&self, f: &mut Frame) {
        match self {
            Screen::Menu(s) => menu::draw(f, s),
            Screen::Config(s) => config::draw(f, s),
            Screen::Training(s) => training::draw(f, s),
        }
    }

    pub fn handle_key(&mut self, key: KeyCode) -> Action {
        match self {
            Screen::Menu(s) => menu::handle_key(s, key),
            Screen::Config(s) => config::handle_key(s, key),
            Screen::Training(s) => training::handle_key(s, key),
        }
    }
}