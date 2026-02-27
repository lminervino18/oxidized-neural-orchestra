pub mod menu;

use crossterm::event::KeyCode;
use ratatui::Frame;

pub enum Action {
    None,
    Quit,
    Transition(Screen),
}

pub enum Screen {
    Menu(menu::MenuState),
}

impl Screen {
    pub fn draw(&self, f: &mut Frame) {
        match self {
            Screen::Menu(s) => menu::draw(f, s),
        }
    }

    pub fn handle_key(&mut self, key: KeyCode) -> Action {
        match self {
            Screen::Menu(s) => menu::handle_key(s, key),
        }
    }
}