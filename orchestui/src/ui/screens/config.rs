use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::ui::utils::centered_rect;
use crate::ui::{screens::training::TrainingState, theme::Theme};
use crate::{config::json, ui::screens::menu::MenuState};

use super::{Action, Screen};

const DEFAULT_MODEL_PATH: &str = "model.json";
const DEFAULT_TRAINING_PATH: &str = "training.json";

const EXAMPLE_MODEL: &str = concat!(
    "{\n",
    "  \"layers\": [\n",
    "    {\n",
    "      \"dense\": {\n",
    "        \"output_size\": 4,\n",
    "        \"init\": \"kaiming\",\n",
    "        \"act_fn\": { \"sigmoid\": { \"amp\": 1.0 } }\n",
    "      }\n",
    "    },\n",
    "    {\n",
    "      \"dense\": {\n",
    "        \"output_size\": 1,\n",
    "        \"init\": { \"const\": { \"value\": 0.0 } },\n",
    "        \"act_fn\": null\n",
    "      }\n",
    "    }\n",
    "  ]\n",
    "}\n",
    "\n",
    "layer types: dense | conv\n",
    "init values: const, uniform, uniform_inclusive,\n",
    "  xavier_uniform, lecun_uniform, normal,\n",
    "  kaiming, xavier, lecun\n",
    "act_fn values: { \"sigmoid\": { \"amp\": 1.0 } }\n",
    "  set to null to disable\n",
    "\n",
    "conv example:\n",
    "  { \"conv\": { \"input_dim\": [1, 28, 28],\n",
    "    \"kernel_dim\": [32, 1, 3], \"stride\": 1,\n",
    "    \"padding\": 1, \"init\": \"kaiming\" } }",
);

const EXAMPLE_TRAINING: &str = concat!(
    "{\n",
    "  \"addrs\": [\"node-0:40000\", \"node-1:40001\", \"node-2:40002\"],\n",
    "  \"algorithm\": {\n",
    "    \"parameter_server\": {\n",
    "      \"nservers\": 1,\n",
    "      \"synchronizer\": \"non_blocking\",\n",
    "      \"store\": \"blocking\"\n",
    "    }\n",
    "  },\n",
    "  \"serializer\": { \"sparse_capable\": { \"r\": 0.95 } },\n",
    "  \"dataset\": {\n",
    "    \"src\": { \"local\": { \"samples_path\": \"x.bin\", \"labels_path\": \"y.bin\" } },\n",
    "    \"x_size\": 2,\n",
    "    \"y_size\": 1\n",
    "  },\n",
    "  \"optimizer\": { \"gradient_descent\": { \"lr\": 0.01 } },\n",
    "  \"loss_fn\": \"mse\",\n",
    "  \"batch_size\": 32,\n",
    "  \"max_epochs\": 100,\n",
    "  \"offline_epochs\": 0,\n",
    "  \"seed\": null\n",
    "}\n",
    "\n",
    "algorithm: \"all_reduce\" | { \"parameter_server\": {..} }\n",
    "  | { \"strategy_switch\": {..} }\n",
    "nservers: how many of the addrs become servers\n",
    "serializer: \"base\" | { \"sparse_capable\": { \"r\": 0.0..1.0 } }\n",
    "synchronizer: \"barrier\" | \"non_blocking\"\n",
    "store: \"blocking\" | \"wild\"\n",
    "optimizer types: gradient_descent",
);

#[derive(Debug, Clone, PartialEq)]
enum Step {
    ModelPath,
    TrainingPath,
    ExampleModel,
    ExampleTraining,
    InvalidConfig { reason: String },
}

/// State for the configuration screen.
pub struct ConfigState {
    step: Step,
    model_path: String,
    model_cursor: usize,
    training_path: String,
    training_cursor: usize,
}

impl ConfigState {
    /// Creates a new `ConfigState` at the first input step.
    pub fn new() -> Self {
        Self {
            step: Step::ModelPath,
            model_path: String::new(),
            model_cursor: 0,
            training_path: String::new(),
            training_cursor: 0,
        }
    }
}

/// Handles a key event for the configuration screen.
///
/// # Args
/// * `state` - The current configuration screen state.
/// * `key` - The key that was pressed.
///
/// # Returns
/// `Some(Action)` if the application state should change, `None` otherwise.
pub fn handle_key(state: &mut ConfigState, key: KeyCode) -> Option<Action> {
    match state.step.clone() {
        Step::ModelPath => handle_model_path(state, key),
        Step::TrainingPath => handle_training_path(state, key),
        Step::ExampleModel | Step::ExampleTraining => {
            state.step = Step::ModelPath;
            None
        }
        Step::InvalidConfig { .. } => match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                Some(Action::Transition(Box::new(Screen::Menu(MenuState::new()))))
            }
            _ => {
                state.step = Step::ModelPath;
                None
            }
        },
    }
}

fn handle_model_path(state: &mut ConfigState, key: KeyCode) -> Option<Action> {
    match key {
        KeyCode::Char('?') => {
            state.step = Step::ExampleModel;
            None
        }
        KeyCode::Tab | KeyCode::End => {
            let ghost = path_ghost(&state.model_path, DEFAULT_MODEL_PATH);
            if !ghost.is_empty() {
                state.model_path.push_str(ghost);
                state.model_cursor = state.model_path.len();
            } else if key == KeyCode::End {
                state.model_cursor = state.model_path.len();
            }
            None
        }
        KeyCode::Home => {
            state.model_cursor = 0;
            None
        }
        KeyCode::Left => {
            state.model_cursor = state.model_cursor.saturating_sub(1);
            None
        }
        KeyCode::Right => {
            if state.model_cursor < state.model_path.len() {
                state.model_cursor += 1;
            } else {
                let ghost = path_ghost(&state.model_path, DEFAULT_MODEL_PATH);
                if !ghost.is_empty() {
                    state.model_path.push_str(ghost);
                    state.model_cursor = state.model_path.len();
                }
            }
            None
        }
        KeyCode::Char(c) => {
            state.model_path.insert(state.model_cursor, c);
            state.model_cursor += 1;
            None
        }
        KeyCode::Backspace => {
            if state.model_cursor > 0 {
                state.model_path.remove(state.model_cursor - 1);
                state.model_cursor -= 1;
            }
            None
        }
        KeyCode::Enter => {
            state.step = Step::TrainingPath;
            None
        }
        KeyCode::Esc => Some(Action::Transition(Box::new(Screen::Menu(MenuState::new())))),
        _ => None,
    }
}

fn handle_training_path(state: &mut ConfigState, key: KeyCode) -> Option<Action> {
    match key {
        KeyCode::Char('?') => {
            state.step = Step::ExampleTraining;
            None
        }
        KeyCode::Tab | KeyCode::End => {
            let ghost = path_ghost(&state.training_path, DEFAULT_TRAINING_PATH);
            if !ghost.is_empty() {
                state.training_path.push_str(ghost);
                state.training_cursor = state.training_path.len();
            } else if key == KeyCode::End {
                state.training_cursor = state.training_path.len();
            }
            None
        }
        KeyCode::Home => {
            state.training_cursor = 0;
            None
        }
        KeyCode::Left => {
            state.training_cursor = state.training_cursor.saturating_sub(1);
            None
        }
        KeyCode::Right => {
            if state.training_cursor < state.training_path.len() {
                state.training_cursor += 1;
            } else {
                let ghost = path_ghost(&state.training_path, DEFAULT_TRAINING_PATH);
                if !ghost.is_empty() {
                    state.training_path.push_str(ghost);
                    state.training_cursor = state.training_path.len();
                }
            }
            None
        }
        KeyCode::Char(c) => {
            state.training_path.insert(state.training_cursor, c);
            state.training_cursor += 1;
            None
        }
        KeyCode::Backspace => {
            if state.training_cursor > 0 {
                state.training_path.remove(state.training_cursor - 1);
                state.training_cursor -= 1;
            }
            None
        }
        KeyCode::Enter => try_load(state),
        KeyCode::Esc => {
            state.step = Step::ModelPath;
            None
        }
        _ => None,
    }
}

/// Returns the ghost-text suffix to suggest. When the input is empty it offers
/// the full default path; when it ends with '/' it offers just the default file
/// name to append after the typed directory; otherwise nothing.
fn path_ghost<'a>(current: &str, default_path: &'a str) -> &'a str {
    if current.is_empty() {
        default_path
    } else if current.ends_with('/') {
        default_path.rsplit('/').next().unwrap_or(default_path)
    } else {
        ""
    }
}

fn try_load(state: &mut ConfigState) -> Option<Action> {
    let model_path = if state.model_path.trim().is_empty() {
        DEFAULT_MODEL_PATH
    } else {
        state.model_path.trim()
    };

    let training_path = if state.training_path.trim().is_empty() {
        DEFAULT_TRAINING_PATH
    } else {
        state.training_path.trim()
    };

    let model_json = match json::load_model(model_path) {
        Ok(d) => d,
        Err(e) => {
            state.step = Step::InvalidConfig {
                reason: format!("{model_path}: {e}"),
            };
            return None;
        }
    };

    let training_json = match json::load_training(training_path) {
        Ok(d) => d,
        Err(e) => {
            state.step = Step::InvalidConfig {
                reason: format!("{training_path}: {e}"),
            };
            return None;
        }
    };

    Some(Action::Transition(Box::new(Screen::Training(Box::new(
        TrainingState::new(
            model_json.config,
            training_json.config,
            training_json.worker_count,
            training_json.server_count,
        ),
    )))))
}

/// Draws the configuration screen.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `state` - The current configuration screen state.
pub fn draw(f: &mut Frame, state: &ConfigState) {
    let area = f.size();
    f.render_widget(Block::default().style(Theme::base()), area);

    match &state.step {
        Step::ModelPath => draw_path_input(
            f,
            area,
            "Model Configuration",
            "model.json path",
            &state.model_path,
            state.model_cursor,
            DEFAULT_MODEL_PATH,
            "Step 1 of 2",
        ),
        Step::TrainingPath => draw_path_input(
            f,
            area,
            "Training Configuration",
            "training.json path",
            &state.training_path,
            state.training_cursor,
            DEFAULT_TRAINING_PATH,
            "Step 2 of 2",
        ),
        Step::ExampleModel => draw_example(f, area, "model.json — example", EXAMPLE_MODEL),
        Step::ExampleTraining => draw_example(f, area, "training.json — example", EXAMPLE_TRAINING),
        Step::InvalidConfig { reason } => draw_invalid_config(f, area, reason),
    }
}

fn draw_invalid_config(f: &mut Frame, area: Rect, reason: &str) {
    let outer = centered_rect(60, 60, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(4),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .split(outer);

    f.render_widget(
        Paragraph::new(Span::styled(
            "Invalid Configuration",
            Theme::error().add_modifier(Modifier::BOLD),
        )),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(Span::styled(
            "Please fix your JSON files and try again.",
            Theme::muted(),
        )),
        chunks[1],
    );

    f.render_widget(
        Paragraph::new(reason)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Theme::error())
                    .title(" Reason ")
                    .title_style(Theme::error()),
            )
            .style(Theme::text())
            .wrap(Wrap { trim: true }),
        chunks[3],
    );

    render_hints(
        f,
        chunks[5],
        &[
            ("any key", "edit config files"),
            ("q / esc", "back to menu"),
        ],
    );
}

fn draw_path_input(
    f: &mut Frame,
    area: Rect,
    title: &str,
    label: &str,
    current: &str,
    cursor: usize,
    default: &str,
    step_label: &str,
) {
    let outer = centered_rect(55, 70, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(5),
        ])
        .split(outer);

    f.render_widget(
        Paragraph::new(Span::styled(
            title,
            Theme::title().add_modifier(Modifier::BOLD),
        )),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(Span::styled(step_label, Theme::dim())),
        chunks[1],
    );

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border())
        .title(format!(" {label} "))
        .title_style(Theme::title());

    let inner = input_block.inner(chunks[3]);
    f.render_widget(input_block, chunks[3]);

    // Split text at cursor to place the block cursor correctly.
    let cursor = cursor.min(current.len());
    let before = &current[..cursor];
    let after  = &current[cursor..];
    let ghost  = path_ghost(current, default);

    let display = if current.is_empty() {
        Line::from(vec![
            Span::styled("█", Theme::accent_cyan()),
            Span::styled(ghost, Theme::muted()),
        ])
    } else {
        let mut spans = vec![
            Span::styled(before.to_string(), Theme::ok()),
            Span::styled("█", Theme::accent_cyan()),
        ];
        if !after.is_empty() {
            spans.push(Span::styled(after.to_string(), Theme::ok()));
        }
        if !ghost.is_empty() {
            spans.push(Span::styled(ghost, Theme::muted()));
        }
        Line::from(spans)
    };

    f.render_widget(Paragraph::new(display), inner);

    let hint_line = if ghost.is_empty() {
        format!("leave empty to use ./{default}")
    } else {
        format!("[tab] or [→] to accept  ·  leave empty to use ./{default}")
    };

    f.render_widget(
        Paragraph::new(Span::styled(hint_line, Theme::dim())),
        chunks[5],
    );

    render_hints(
        f,
        chunks[7],
        &[
            ("enter", "confirm"),
            ("←/→", "move cursor"),
            ("tab/→", "accept suggestion"),
            ("?", "view example"),
            ("esc", "back"),
        ],
    );
}

fn draw_example(f: &mut Frame, area: Rect, title: &str, content: &str) {
    let outer = centered_rect(65, 88, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(outer);

    f.render_widget(
        Paragraph::new(Span::styled(
            title,
            Theme::title().add_modifier(Modifier::BOLD),
        )),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(content)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Theme::border()),
            )
            .style(Theme::dim())
            .wrap(Wrap { trim: false }),
        chunks[1],
    );

    render_hints(f, chunks[2], &[("any key", "back")]);
}

fn render_hints(f: &mut Frame, area: Rect, hints: &[(&str, &str)]) {
    let key_col_width = hints
        .iter()
        .map(|(k, _)| k.len() as u16 + 2)
        .max()
        .unwrap_or(8)
        + 2;

    let outer = centered_rect(40, 100, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            hints
                .iter()
                .map(|_| Constraint::Length(1))
                .chain(std::iter::once(Constraint::Min(0)))
                .collect::<Vec<_>>(),
        )
        .split(outer);

    for (i, (key, action)) in hints.iter().enumerate() {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(key_col_width), Constraint::Min(0)])
            .split(rows[i]);

        f.render_widget(
            Paragraph::new(Span::styled(format!("[{key}]"), Theme::accent_cyan())),
            cols[0],
        );
        f.render_widget(Paragraph::new(Span::styled(*action, Theme::dim())), cols[1]);
    }
}
