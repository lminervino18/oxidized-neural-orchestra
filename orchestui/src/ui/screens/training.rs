use std::time::Instant;

use crossterm::event::KeyCode;
use orchestrator::{
    configs::{ModelConfig, TrainingConfig},
    TrainingEvent,
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::Color,
    widgets::Block,
    Frame,
};
use tokio::sync::mpsc;

use crate::ui::components::{
    confirm_quit::draw_confirm_quit, header::draw_header, log_panel::draw_log,
    loss_chart::draw_charts, params_panel::draw_params, workers_table::draw_workers_table,
};
use crate::ui::theme::Theme;

use super::Action;

/// Per-worker colors for charts and table highlights.
pub const WORKER_COLORS: &[Color] = &[
    Color::Rgb(57, 255, 20),
    Color::Rgb(0, 255, 255),
    Color::Rgb(255, 0, 255),
    Color::Rgb(255, 255, 0),
    Color::Rgb(255, 130, 0),
];

/// The current phase of the training session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Waiting for all workers and servers to connect.
    Connecting,
    /// Training is in progress.
    Training,
    /// Training completed successfully.
    Finished,
    /// A fatal error occurred.
    Error,
}

/// Whether the quit-confirmation popup is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmQuit {
    Hidden,
    Visible,
}

/// Live state for a single worker node.
#[derive(Debug, Clone)]
pub struct WorkerState {
    /// Worker index.
    pub id: usize,
    /// Number of epochs completed so far.
    pub epochs_done: usize,
    /// Most recently reported loss value.
    pub last_loss: Option<f32>,
    /// Whether the worker has disconnected.
    pub done: bool,
}

/// Severity level for log entries.
#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Info,
    #[allow(dead_code)]
    Warn,
    Error,
}

/// Full state for the training dashboard screen.
pub struct TrainingState {
    pub workers_total: usize,
    pub servers_total: usize,
    pub optimizer_label: String,
    pub max_epochs: usize,
    pub phase: Phase,
    pub started_at: Instant,
    pub workers: Vec<WorkerState>,
    /// Per-worker loss time series as (epoch, loss) pairs.
    pub loss_series: Vec<Vec<(f64, f64)>>,
    pub logs: Vec<(LogLevel, String)>,
    pub final_params: Option<Vec<f32>>,
    pub error: Option<String>,
    pub events: mpsc::Receiver<TrainingEvent>,
    pub confirm_quit: ConfirmQuit,
    pub selected_worker: usize,
    /// Set when the session fails to start — shown as a full-screen error.
    pub startup_error: Option<String>,
}

impl TrainingState {
    /// Creates a new `TrainingState`, starting the training session immediately.
    ///
    /// If the session fails to start, the state transitions to an error screen
    /// rather than panicking.
    ///
    /// # Args
    /// * `model` - The model architecture configuration.
    /// * `training` - The training configuration.
    /// * `workers_total` - The number of worker nodes expected.
    /// * `servers_total` - The number of parameter servers expected.
    pub fn new(
        model: ModelConfig,
        training: TrainingConfig<String>,
        workers_total: usize,
        servers_total: usize,
    ) -> Self {
        let optimizer_label = format!("{:?}", training.optimizer)
            .split_whitespace()
            .next()
            .unwrap_or("unknown")
            .to_string();

        let max_epochs = training.max_epochs.get();

        let session = match orchestrator::train(model, training) {
            Ok(s) => s,
            Err(e) => {
                return Self::dead(
                    workers_total,
                    servers_total,
                    optimizer_label,
                    max_epochs,
                    e.to_string(),
                );
            }
        };

        let events = session.event_listener();

        let workers = (0..workers_total)
            .map(|id| WorkerState {
                id,
                epochs_done: 0,
                last_loss: None,
                done: false,
            })
            .collect();

        let loss_series = vec![Vec::new(); workers_total];

        Self {
            workers_total,
            servers_total,
            optimizer_label,
            max_epochs,
            phase: Phase::Connecting,
            started_at: Instant::now(),
            workers,
            loss_series,
            logs: vec![(
                LogLevel::Info,
                format!("connecting to {workers_total} worker(s) and {servers_total} server(s)..."),
            )],
            final_params: None,
            error: None,
            events,
            confirm_quit: ConfirmQuit::Hidden,
            selected_worker: 0,
            startup_error: None,
        }
    }

    /// Creates a dead state used when the session fails to start.
    fn dead(
        workers_total: usize,
        servers_total: usize,
        optimizer_label: String,
        max_epochs: usize,
        err: String,
    ) -> Self {
        let (_, rx) = mpsc::channel(1);
        Self {
            workers_total,
            servers_total,
            optimizer_label,
            max_epochs,
            phase: Phase::Error,
            started_at: Instant::now(),
            workers: Vec::new(),
            loss_series: Vec::new(),
            logs: Vec::new(),
            final_params: None,
            error: Some(err.clone()),
            events: rx,
            confirm_quit: ConfirmQuit::Hidden,
            selected_worker: 0,
            startup_error: Some(err),
        }
    }

    /// Drains all pending training events and applies them to the state.
    pub fn tick(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            self.apply(event);
        }
    }

    /// Applies a single training event to the state.
    fn apply(&mut self, event: TrainingEvent) {
        match event {
            TrainingEvent::Loss { worker_id, losses } => {
                self.phase = Phase::Training;

                if worker_id < self.workers.len() {
                    for loss in &losses {
                        self.workers[worker_id].epochs_done += 1;
                        self.workers[worker_id].last_loss = Some(*loss);

                        if let Some(series) = self.loss_series.get_mut(worker_id) {
                            let epoch = self.workers[worker_id].epochs_done as f64;
                            series.push((epoch, *loss as f64));
                        }
                    }

                    let epochs_done = self.workers[worker_id].epochs_done;
                    let last = losses.last().copied().unwrap_or(0.0);
                    self.push_log(
                        LogLevel::Info,
                        format!("worker {worker_id}  epoch {epochs_done}  loss={last:.4}"),
                    );
                }
            }

            TrainingEvent::WorkerDone(worker_id) => {
                if worker_id < self.workers.len() {
                    self.workers[worker_id].done = true;
                }
                self.push_log(LogLevel::Info, format!("worker {worker_id} disconnected"));
            }

            TrainingEvent::Complete(params) => {
                self.phase = Phase::Finished;
                self.push_log(
                    LogLevel::Info,
                    format!("training complete — {} parameters received", params.len()),
                );
                self.final_params = Some(params);
            }

            TrainingEvent::Error(e) => {
                self.phase = Phase::Error;
                let msg = e.to_string();
                self.error = Some(msg.clone());
                self.push_log(LogLevel::Error, msg);
            }
        }
    }

    /// Appends a log entry, evicting the oldest if the buffer exceeds 200 entries.
    fn push_log(&mut self, level: LogLevel, msg: String) {
        self.logs.push((level, msg));
        if self.logs.len() > 200 {
            self.logs.remove(0);
        }
    }

    /// Returns the elapsed time since the session started as a `MM:SS` string.
    pub fn elapsed_str(&self) -> String {
        let s = self.started_at.elapsed().as_secs();
        format!("{:02}:{:02}", s / 60, s % 60)
    }

    /// Returns the number of workers that have finished.
    pub fn workers_done(&self) -> usize {
        self.workers.iter().filter(|w| w.done).count()
    }

    /// Returns `true` if the session is still connecting or training.
    pub fn is_active(&self) -> bool {
        matches!(self.phase, Phase::Connecting | Phase::Training)
    }

    /// Computes the average loss series across all workers.
    pub fn avg_loss_series(&self) -> Vec<(f64, f64)> {
        let max_len = self.loss_series.iter().map(|s| s.len()).max().unwrap_or(0);
        if max_len == 0 {
            return Vec::new();
        }

        (0..max_len)
            .filter_map(|i| {
                let values: Vec<f64> = self
                    .loss_series
                    .iter()
                    .filter_map(|s| s.get(i).map(|(_, y)| *y))
                    .collect();

                if values.is_empty() {
                    return None;
                }

                let epoch = self
                    .loss_series
                    .iter()
                    .find_map(|s| s.get(i).map(|(x, _)| *x))
                    .unwrap_or(i as f64 + 1.0);

                Some((epoch, values.iter().sum::<f64>() / values.len() as f64))
            })
            .collect()
    }

    /// Advances the selected worker to the next one (wrapping).
    pub fn next_worker(&mut self) {
        if self.workers_total > 0 {
            self.selected_worker = (self.selected_worker + 1) % self.workers_total;
        }
    }

    /// Moves the selected worker to the previous one (wrapping).
    pub fn prev_worker(&mut self) {
        if self.workers_total > 0 {
            self.selected_worker =
                (self.selected_worker + self.workers_total - 1) % self.workers_total;
        }
    }
}

/// Handles a key event for the training screen.
pub fn handle_key(state: &mut TrainingState, key: KeyCode) -> Option<Action> {
    if state.startup_error.is_some() {
        return Some(Action::Transition(super::Screen::Menu(
            crate::ui::screens::menu::MenuState::new(),
        )));
    }

    match (key, state.confirm_quit, state.is_active()) {
        (KeyCode::Left, ConfirmQuit::Hidden, _) => {
            state.prev_worker();
            None
        }
        (KeyCode::Right, ConfirmQuit::Hidden, _) => {
            state.next_worker();
            None
        }
        (KeyCode::Char('q') | KeyCode::Esc, ConfirmQuit::Hidden, true) => {
            state.confirm_quit = ConfirmQuit::Visible;
            None
        }
        (KeyCode::Char('y') | KeyCode::Char('Y'), ConfirmQuit::Visible, _) => {
            Some(Action::Transition(super::Screen::Menu(
                crate::ui::screens::menu::MenuState::new(),
            )))
        }
        (KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc, ConfirmQuit::Visible, _) => {
            state.confirm_quit = ConfirmQuit::Hidden;
            None
        }
        (KeyCode::Char('q') | KeyCode::Esc, ConfirmQuit::Hidden, false) => {
            Some(Action::Transition(super::Screen::Menu(
                crate::ui::screens::menu::MenuState::new(),
            )))
        }
        _ => None,
    }
}

/// Draws the training dashboard screen.
pub fn draw(f: &mut Frame, state: &mut TrainingState) {
    state.tick();

    let area = f.size();
    f.render_widget(Block::default().style(Theme::base()), area);

    if let Some(err) = &state.startup_error.clone() {
        draw_startup_error(f, area, err);
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(8),
        ])
        .split(area);

    draw_header(f, rows[0], state);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(rows[1]);

    draw_charts(f, body[0], state);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints(if state.final_params.is_some() {
            vec![Constraint::Min(4), Constraint::Length(5)]
        } else {
            vec![Constraint::Min(4)]
        })
        .split(body[1]);

    draw_workers_table(f, right[0], state);

    if state.final_params.is_some() {
        draw_params(f, right[1], state);
    }

    draw_log(f, rows[2], state);

    if state.confirm_quit == ConfirmQuit::Visible {
        draw_confirm_quit(f, area);
    }
}

fn draw_startup_error(f: &mut Frame, area: ratatui::layout::Rect, err: &str) {
    use ratatui::{
        layout::{Alignment, Constraint, Direction, Layout},
        style::Modifier,
        text::Span,
        widgets::{Block, Borders, Paragraph, Wrap},
    };

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Min(0),
            Constraint::Percentage(20),
        ])
        .split(area)[1];

    let outer = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(15),
            Constraint::Min(0),
            Constraint::Percentage(15),
        ])
        .split(outer)[1];

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(4),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(outer);

    f.render_widget(
        Paragraph::new(Span::styled(
            "Failed to Start Training",
            Theme::error().add_modifier(Modifier::BOLD),
        )),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(Span::styled(
            "Could not connect to workers or servers. Please check your JSON files and try again.",
            Theme::muted(),
        ))
        .wrap(Wrap { trim: true }),
        chunks[1],
    );

    f.render_widget(
        Paragraph::new(err)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Theme::error())
                    .title(" Error ")
                    .title_style(Theme::error().add_modifier(Modifier::BOLD)),
            )
            .style(Theme::text())
            .wrap(Wrap { trim: true }),
        chunks[3],
    );

    f.render_widget(
        Paragraph::new(Span::styled(
            "Press any key to go back to the menu.",
            Theme::muted(),
        ))
        .alignment(Alignment::Center),
        chunks[5],
    );
}
