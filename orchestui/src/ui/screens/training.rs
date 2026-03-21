use std::time::Instant;

use crossterm::event::KeyCode;
use orchestrator::{
    Session,
    configs::{DatasetSrc, ModelConfig, TrainingConfig},
    TrainedModel, TrainingEvent,
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::Color,
    widgets::Block,
    Frame,
};

use super::{Action, Screen};
use crate::ui::{
    components::{
        confirm_quit::draw_confirm_quit,
        converting::draw_converting,
        header::draw_header,
        log_panel::draw_log,
        loss_chart::draw_charts,
        params_panel::draw_params,
        save_popup::draw_save_popup,
        workers_table::draw_workers_table,
    },
    screens::menu::MenuState,
    theme::Theme,
    utils::fmt_loss,
};

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
    /// Dataset is being converted from a delimited format to binary.
    Converting,
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

/// Whether the save-path popup is shown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SavePopup {
    /// Popup is not visible.
    Hidden,
    /// Popup is visible, holding the current path input.
    Visible(String),
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

/// Internal startup result sent from the background thread.
enum StartupResult {
    Ok(Session),
    Err(String),
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
    /// The trained model received on completion, including architecture and params.
    pub final_trained: Option<TrainedModel>,
    pub error: Option<String>,
    /// Active training event receiver, available once the session starts.
    pub events: Option<tokio::sync::mpsc::Receiver<TrainingEvent>>,
    pub confirm_quit: ConfirmQuit,
    pub save_popup: SavePopup,
    pub selected_worker: usize,
    /// Set when the session fails to start — shown as a full-screen error.
    pub startup_error: Option<String>,
    /// Channel that receives the session once `train()` completes in the background.
    startup_rx: Option<std::sync::mpsc::Receiver<StartupResult>>,
}

impl TrainingState {
    /// Creates a new `TrainingState`, spawning `train()` in a background thread
    /// so the TUI remains responsive during dataset conversion and connection setup.
    ///
    /// Starts in `Phase::Converting` only when the dataset source is a delimited
    /// file that requires conversion. Otherwise starts in `Phase::Connecting`.
    /// Transitions to `Phase::Error` if the background thread fails.
    ///
    /// # Args
    /// * `model` - The model architecture configuration.
    /// * `training` - The training configuration.
    /// * `workers_total` - The number of worker nodes expected.
    /// * `servers_total` - The number of parameter servers expected.
    pub fn new(
        model: ModelConfig,
        training: TrainingConfig,
        workers_total: usize,
        servers_total: usize,
    ) -> Self {
        let optimizer_label = format!("{:?}", training.optimizer)
            .split_whitespace()
            .next()
            .unwrap_or("unknown")
            .to_string();

        let max_epochs = training.max_epochs.get();

        let initial_phase = match &training.dataset.src {
            DatasetSrc::Local { path }
                if orchestrator::dataset_format::DatasetFormat::from_path(path).is_some() =>
            {
                Phase::Converting
            }
            _ => Phase::Connecting,
        };

        let (startup_tx, startup_rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let result = match orchestrator::train(model, training) {
                Ok(session) => StartupResult::Ok(session),
                Err(e) => StartupResult::Err(e.to_string()),
            };
            let _ = startup_tx.send(result);
        });

        let workers = (0..workers_total)
            .map(|id| WorkerState {
                id,
                epochs_done: 0,
                last_loss: None,
                done: false,
            })
            .collect();

        let loss_series = vec![Vec::new(); workers_total];

        let initial_log = match initial_phase {
            Phase::Converting => "converting dataset to binary format...",
            _ => "connecting to workers and servers...",
        };

        Self {
            workers_total,
            servers_total,
            optimizer_label,
            max_epochs,
            phase: initial_phase,
            started_at: Instant::now(),
            workers,
            loss_series,
            logs: vec![(LogLevel::Info, initial_log.into())],
            final_trained: None,
            error: None,
            events: None,
            confirm_quit: ConfirmQuit::Hidden,
            save_popup: SavePopup::Hidden,
            selected_worker: 0,
            startup_error: None,
            startup_rx: Some(startup_rx),
        }
    }

    /// Drains all pending training events and applies them to the state.
    ///
    /// Also checks whether the background startup thread has finished and,
    /// if so, transitions to `Phase::Connecting` and starts listening for
    /// training events, or to `Phase::Error` on failure.
    pub fn tick(&mut self) {
        let startup_result = self
            .startup_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());

        if let Some(result) = startup_result {
            self.startup_rx = None;
            match result {
                StartupResult::Ok(session) => {
                    self.push_log(
                        LogLevel::Info,
                        format!(
                            "connecting to {} worker(s) and {} server(s)...",
                            self.workers_total, self.servers_total
                        ),
                    );
                    self.phase = Phase::Connecting;
                    self.events = Some(session.event_listener());
                }
                StartupResult::Err(e) => {
                    self.phase = Phase::Error;
                    self.startup_error = Some(e.clone());
                    self.error = Some(e);
                }
            }
        }

        let events: Vec<TrainingEvent> = self
            .events
            .as_mut()
            .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect())
            .unwrap_or_default();

        for event in events {
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
                        format!(
                            "worker {worker_id}  epoch {epochs_done}  loss={}",
                            fmt_loss(last)
                        ),
                    );
                }
            }

            TrainingEvent::WorkerDone(worker_id) => {
                if worker_id < self.workers.len() {
                    self.workers[worker_id].done = true;
                }
                self.push_log(LogLevel::Info, format!("worker {worker_id} disconnected"));
            }

            TrainingEvent::Complete(trained) => {
                self.phase = Phase::Finished;
                self.push_log(
                    LogLevel::Info,
                    format!(
                        "training complete — {} parameters received",
                        trained.params().len()
                    ),
                );
                self.final_trained = Some(trained);
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

    /// Returns `true` if the session is still converting, connecting, or training.
    pub fn is_active(&self) -> bool {
        matches!(
            self.phase,
            Phase::Converting | Phase::Connecting | Phase::Training
        )
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
        return Some(Action::Transition(Box::new(Screen::Menu(MenuState::new()))));
    }

    if let SavePopup::Visible(ref mut path) = state.save_popup {
        match key {
            KeyCode::Char(c) => {
                path.push(c);
                return None;
            }
            KeyCode::Backspace => {
                path.pop();
                return None;
            }
            KeyCode::Esc => {
                state.save_popup = SavePopup::Hidden;
                return None;
            }
            KeyCode::Enter => {
                let resolved = if path.trim().is_empty() {
                    "model.safetensors".to_string()
                } else {
                    path.trim().to_string()
                };
                state.save_popup = SavePopup::Hidden;
                if let Some(trained) = &state.final_trained {
                    match trained.save_safetensors(&resolved) {
                        Ok(()) => {
                            state.push_log(LogLevel::Info, format!("model saved to {resolved}"))
                        }
                        Err(e) => state.push_log(LogLevel::Error, format!("failed to save: {e}")),
                    }
                }
                return None;
            }
            _ => return None,
        }
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
        (KeyCode::Char('s'), ConfirmQuit::Hidden, false) if state.final_trained.is_some() => {
            state.save_popup = SavePopup::Visible(String::new());
            None
        }
        (KeyCode::Char('q') | KeyCode::Esc, ConfirmQuit::Hidden, true) => {
            state.confirm_quit = ConfirmQuit::Visible;
            None
        }
        (KeyCode::Char('y') | KeyCode::Char('Y'), ConfirmQuit::Visible, _) => {
            Some(Action::Transition(Box::new(Screen::Menu(MenuState::new()))))
        }
        (KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc, ConfirmQuit::Visible, _) => {
            state.confirm_quit = ConfirmQuit::Hidden;
            None
        }
        (KeyCode::Char('q') | KeyCode::Esc, ConfirmQuit::Hidden, false) => {
            Some(Action::Transition(Box::new(Screen::Menu(MenuState::new()))))
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
        .constraints(if state.final_trained.is_some() {
            vec![Constraint::Min(4), Constraint::Length(5)]
        } else {
            vec![Constraint::Min(4)]
        })
        .split(body[1]);

    draw_workers_table(f, right[0], state);

    if state.final_trained.is_some() {
        draw_params(f, right[1], state);
    }

    draw_log(f, rows[2], state);

    if state.phase == Phase::Converting {
        draw_converting(f, area, state);
    }

    if state.confirm_quit == ConfirmQuit::Visible {
        draw_confirm_quit(f, area);
    }

    if let SavePopup::Visible(ref path) = state.save_popup {
        draw_save_popup(f, path);
    }
}

fn draw_startup_error(f: &mut Frame, area: ratatui::layout::Rect, err: &str) {
    use ratatui::{
        layout::Alignment,
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