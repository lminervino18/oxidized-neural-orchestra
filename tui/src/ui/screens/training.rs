use crossterm::event::KeyCode;
use orchestrator::configs::{ModelConfig, TrainingConfig};
use ratatui::{
    Frame,
    layout::Alignment,
    text::Span,
    widgets::{Block, Paragraph},
};

use crate::ui::theme::Theme;

use super::Action;

pub struct TrainingState {
    pub model: ModelConfig,
    pub training: TrainingConfig<String>,
    pub workers_total: usize,
}

impl TrainingState {
    pub fn new(
        model: ModelConfig,
        training: TrainingConfig<String>,
        workers_total: usize,
    ) -> Self {
        Self { model, training, workers_total }
    }
}

pub fn handle_key(_state: &mut TrainingState, key: KeyCode) -> Action {
    match key {
        KeyCode::Char('q') | KeyCode::Esc => Action::Transition(super::Screen::Menu(
            crate::ui::screens::menu::MenuState::new(),
        )),
        _ => Action::None,
    }
}

pub fn draw(f: &mut Frame, _state: &TrainingState) {
    let area = f.size();
    f.render_widget(Block::default().style(Theme::base()), area);
    f.render_widget(
        Paragraph::new(Span::styled(
            "Training starting... (step 3)",
            Theme::title(),
        ))
        .alignment(Alignment::Center),
        area,
    );
}