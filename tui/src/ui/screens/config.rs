use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::config::{builder, json};
use crate::ui::theme::Theme;

use super::{Action, Screen};

const DEFAULT_MODEL_PATH: &str = "model.json";
const DEFAULT_TRAINING_PATH: &str = "training.json";

const EXAMPLE_MODEL: &str = concat!(
    "{\n",
    "  \"layers\": [\n",
    "    {\n",
    "      \"n\": 2, \"m\": 4,\n",
    "      \"init\": \"xavier_uniform\",\n",
    "      \"act_fn\": \"sigmoid\", \"act_amp\": 1.0\n",
    "    },\n",
    "    {\n",
    "      \"n\": 4, \"m\": 1,\n",
    "      \"init\": \"const\", \"init_value\": 0.0,\n",
    "      \"act_fn\": null\n",
    "    }\n",
    "  ]\n",
    "}\n",
    "\n",
    "init values: const, uniform, uniform_inclusive,\n",
    "  xavier_uniform, lecun_uniform, normal,\n",
    "  kaiming, xavier, lecun\n",
    "act_fn values: sigmoid, null",
);

const EXAMPLE_TRAINING: &str = concat!(
    "{\n",
    "  \"worker_addrs\": [\"worker-0:50000\"],\n",
    "  \"server_addrs\": [\"server-0:40000\"],\n",
    "  \"synchronizer\": \"barrier\",\n",
    "  \"barrier_size\": 1,\n",
    "  \"store\": \"blocking\",\n",
    "  \"shard_size\": 128,\n",
    "  \"max_epochs\": 100,\n",
    "  \"offline_epochs\": 0,\n",
    "  \"batch_size\": 32,\n",
    "  \"seed\": null,\n",
    "  \"optimizer\": \"gradient_descent\",\n",
    "  \"lr\": 0.01\n",
    "}\n",
    "\n",
    "synchronizer: barrier, non_blocking\n",
    "store: blocking, wild\n",
    "optimizer: gradient_descent, adam,\n",
    "  gradient_descent_with_momentum",
);

#[derive(Debug, Clone, PartialEq)]
enum Step {
    ModelPath,
    TrainingPath,
    ExampleModel,
    ExampleTraining,
    /// Config loaded but invalid — show error and let user go back to fix JSONs.
    InvalidConfig { reason: String },
}

pub struct ConfigState {
    step: Step,
    model_path: String,
    training_path: String,
    pub error: Option<String>,
}

impl ConfigState {
    pub fn new() -> Self {
        Self {
            step: Step::ModelPath,
            model_path: String::new(),
            training_path: String::new(),
            error: None,
        }
    }
}

pub fn handle_key(state: &mut ConfigState, key: KeyCode) -> Action {
    state.error = None;

    match state.step.clone() {
        Step::ModelPath => handle_model_path(state, key),
        Step::TrainingPath => handle_training_path(state, key),
        Step::ExampleModel | Step::ExampleTraining => {
            state.step = Step::ModelPath;
            Action::None
        }
        Step::InvalidConfig { .. } => match key {
            // q or esc → back to menu
            KeyCode::Char('q') | KeyCode::Esc => {
                Action::Transition(Screen::Menu(crate::ui::screens::menu::MenuState::new()))
            }
            // any other key → back to path input to fix JSONs
            _ => {
                state.step = Step::ModelPath;
                Action::None
            }
        },
    }
}

fn handle_model_path(state: &mut ConfigState, key: KeyCode) -> Action {
    match key {
        KeyCode::Char('?') => {
            state.step = Step::ExampleModel;
            Action::None
        }
        KeyCode::Char(c) => {
            state.model_path.push(c);
            Action::None
        }
        KeyCode::Backspace => {
            state.model_path.pop();
            Action::None
        }
        KeyCode::Enter => {
            state.step = Step::TrainingPath;
            Action::None
        }
        KeyCode::Esc => {
            Action::Transition(Screen::Menu(crate::ui::screens::menu::MenuState::new()))
        }
        _ => Action::None,
    }
}

fn handle_training_path(state: &mut ConfigState, key: KeyCode) -> Action {
    match key {
        KeyCode::Char('?') => {
            state.step = Step::ExampleTraining;
            Action::None
        }
        KeyCode::Char(c) => {
            state.training_path.push(c);
            Action::None
        }
        KeyCode::Backspace => {
            state.training_path.pop();
            Action::None
        }
        KeyCode::Enter => try_load(state),
        KeyCode::Esc => {
            state.step = Step::ModelPath;
            Action::None
        }
        _ => Action::None,
    }
}

fn try_load(state: &mut ConfigState) -> Action {
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

    let model_draft = match json::load_model(model_path) {
        Ok(d) => d,
        Err(e) => {
            state.error = Some(format!("model.json: {e}"));
            state.step = Step::ModelPath;
            return Action::None;
        }
    };

    let training_draft = match json::load_training(training_path) {
        Ok(d) => d,
        Err(e) => {
            state.error = Some(format!("training.json: {e}"));
            return Action::None;
        }
    };

    match builder::build(&model_draft, &training_draft) {
        Ok((model, training)) => {
            let workers_total = training_draft.worker_addrs.len();
            Action::Transition(Screen::Training(
                crate::ui::screens::training::TrainingState::new(model, training, workers_total),
            ))
        }
        Err(reason) => {
            state.step = Step::InvalidConfig { reason };
            Action::None
        }
    }
}

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
            DEFAULT_MODEL_PATH,
            "Step 1 of 2",
        ),
        Step::TrainingPath => draw_path_input(
            f,
            area,
            "Training Configuration",
            "training.json path",
            &state.training_path,
            DEFAULT_TRAINING_PATH,
            "Step 2 of 2",
        ),
        Step::ExampleModel => draw_example(f, area, "model.json — example", EXAMPLE_MODEL),
        Step::ExampleTraining => {
            draw_example(f, area, "training.json — example", EXAMPLE_TRAINING)
        }
        Step::InvalidConfig { reason } => draw_invalid_config(f, area, reason),
    }

    if let Some(err) = &state.error {
        draw_error_bar(f, area, err);
    }
}

fn draw_invalid_config(f: &mut Frame, area: Rect, reason: &str) {
    let outer = centered_rect(60, 60, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // title
            Constraint::Length(1), // subtitle
            Constraint::Length(1), // spacer
            Constraint::Min(4),    // reason box
            Constraint::Length(1), // spacer
            Constraint::Length(3), // hints
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
    default: &str,
    step_label: &str,
) {
    let outer = centered_rect(55, 70, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title
            Constraint::Length(1), // step label
            Constraint::Length(2), // spacer
            Constraint::Length(3), // input box
            Constraint::Length(1), // spacer
            Constraint::Length(1), // default note
            Constraint::Min(0),    // spacer
            Constraint::Length(5), // keybinds
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

    let display = if current.is_empty() {
        Line::from(vec![
            Span::styled(default, Theme::muted()),
            Span::styled("█", Theme::accent_cyan()),
        ])
    } else {
        Line::from(vec![
            Span::styled(current, Theme::ok()),
            Span::styled("█", Theme::accent_cyan()),
        ])
    };

    f.render_widget(Paragraph::new(display), inner);

    f.render_widget(
        Paragraph::new(Span::styled(
            format!("leave empty to use ./{default}"),
            Theme::dim(),
        )),
        chunks[5],
    );

    render_hints(
        f,
        chunks[7],
        &[("enter", "confirm"), ("?", "view example"), ("esc", "back")],
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

fn draw_error_bar(f: &mut Frame, area: Rect, msg: &str) {
    let bar = Rect {
        x: area.x + 1,
        y: area.y + area.height - 1,
        width: area.width.saturating_sub(2),
        height: 1,
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ✖ ", Theme::error()),
            Span::styled(msg, Theme::error()),
        ])),
        bar,
    );
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

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vert[1])[1]
}