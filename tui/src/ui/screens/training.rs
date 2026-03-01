use std::time::Instant;

use crossterm::event::KeyCode;
use orchestrator::{TrainingEvent, configs::{ModelConfig, TrainingConfig}};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table, Wrap,
    },
};
use tokio::sync::mpsc;

use crate::ui::theme::Theme;

use super::Action;

// One color per worker, cycling if more than palette size.
const WORKER_COLORS: &[Color] = &[
    Color::Rgb(57, 255, 20),   // neon green
    Color::Rgb(0, 255, 255),   // cyan
    Color::Rgb(255, 0, 255),   // magenta
    Color::Rgb(255, 255, 0),   // yellow
    Color::Rgb(255, 130, 0),   // orange
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Connecting,
    Training,
    Finished,
    Error,
}

#[derive(Debug, Clone)]
struct WorkerState {
    id: usize,
    epochs_done: usize,
    last_loss: Option<f32>,
    done: bool,
}

pub struct TrainingState {
    // Config kept for display
    workers_total: usize,
    optimizer_label: String,
    max_epochs: usize,

    // Live state
    phase: Phase,
    started_at: Instant,
    workers: Vec<WorkerState>,
    // Per-worker loss history as (epoch, loss) for the Chart widget
    loss_series: Vec<Vec<(f64, f64)>>,
    logs: Vec<(LogLevel, String)>,
    final_params: Option<Vec<f32>>,
    error: Option<String>,

    events: mpsc::Receiver<TrainingEvent>,
}

#[derive(Debug, Clone, Copy)]
enum LogLevel {
    Info,
    Warn,
    Error,
}

impl TrainingState {
    pub fn new(
        model: ModelConfig,
        training: TrainingConfig<String>,
        workers_total: usize,
    ) -> Self {
        let optimizer_label = format!("{:?}", training.optimizer)
            .split_whitespace()
            .next()
            .unwrap_or("unknown")
            .to_string();

        let max_epochs = training.max_epochs.get();

        // Start the session — connects and spawns listeners.
        let mut session = match orchestrator::train(model, training) {
            Ok(s) => s,
            Err(e) => {
                // Can't connect: build a dead state with error.
                return Self::dead(workers_total, optimizer_label, max_epochs, e.to_string());
            }
        };

        let events = session.take_events();

        let workers = (0..workers_total)
            .map(|id| WorkerState {
                id,
                epochs_done: 0,
                last_loss: None,
                done: false,
            })
            .collect();

        let loss_series = vec![Vec::new(); workers_total];

        let state = Self {
            workers_total,
            optimizer_label,
            max_epochs,
            phase: Phase::Connecting,
            started_at: Instant::now(),
            workers,
            loss_series,
            logs: vec![(LogLevel::Info, "connecting to workers and parameter server...".into())],
            final_params: None,
            error: None,
            events,
        };

        state
    }

    fn dead(workers_total: usize, optimizer_label: String, max_epochs: usize, err: String) -> Self {
        let (_, rx) = mpsc::channel(1);
        Self {
            workers_total,
            optimizer_label,
            max_epochs,
            phase: Phase::Error,
            started_at: Instant::now(),
            workers: Vec::new(),
            loss_series: Vec::new(),
            logs: vec![(LogLevel::Error, err.clone())],
            final_params: None,
            error: Some(err),
            events: rx,
        }
    }

    /// Drains pending events and updates state. Called every frame.
    pub fn tick(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            self.apply(event);
        }
    }

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

        TrainingEvent::Error(msg) => {
            self.phase = Phase::Error;
            self.error = Some(msg.clone());
            self.push_log(LogLevel::Error, msg);
        }
    }
}

    fn push_log(&mut self, level: LogLevel, msg: String) {
        self.logs.push((level, msg));
        if self.logs.len() > 200 {
            self.logs.remove(0);
        }
    }

    fn elapsed_str(&self) -> String {
        let s = self.started_at.elapsed().as_secs();
        format!("{:02}:{:02}", s / 60, s % 60)
    }

    fn workers_done(&self) -> usize {
        self.workers.iter().filter(|w| w.done).count()
    }
}

pub fn handle_key(state: &mut TrainingState, key: KeyCode) -> Action {
    match key {
        KeyCode::Char('q') | KeyCode::Esc if state.phase == Phase::Finished
            || state.phase == Phase::Error =>
        {
            Action::Transition(super::Screen::Menu(
                crate::ui::screens::menu::MenuState::new(),
            ))
        }
        _ => Action::None,
    }
}

pub fn draw(f: &mut Frame, state: &mut TrainingState) {
    state.tick();

    let area = f.size();
    f.render_widget(Block::default().style(Theme::base()), area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(10),    // body
            Constraint::Length(8),  // log
        ])
        .split(area);

    draw_header(f, rows[0], state);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(65), // chart
            Constraint::Percentage(35), // right panel
        ])
        .split(rows[1]);

    draw_chart(f, body[0], state);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints(if state.final_params.is_some() {
            vec![
                Constraint::Min(4),    // workers table
                Constraint::Length(5), // params panel
            ]
        } else {
            vec![Constraint::Min(4)]
        })
        .split(body[1]);

    draw_workers_table(f, right[0], state);

    if state.final_params.is_some() {
        draw_params(f, right[1], state);
    }

    draw_log(f, rows[2], state);
}

fn draw_header(f: &mut Frame, area: Rect, state: &TrainingState) {
    let phase_span = match state.phase {
        Phase::Connecting => Span::styled("CONNECTING", Theme::accent_cyan()),
        Phase::Training => Span::styled("TRAINING", Theme::ok()),
        Phase::Finished => Span::styled("FINISHED", Theme::accent_magenta()),
        Phase::Error => Span::styled("ERROR", Theme::error()),
    };

    let workers_done = state.workers_done();

    let line = Line::from(vec![
        Span::styled(" ONO  ", Theme::title().add_modifier(Modifier::BOLD)),
        Span::styled("│  ", Theme::muted()),
        phase_span,
        Span::styled("  │  ", Theme::muted()),
        Span::styled(format!("elapsed {}", state.elapsed_str()), Theme::dim()),
        Span::styled("  │  ", Theme::muted()),
        Span::styled(
            format!("workers {}/{}", workers_done, state.workers_total),
            Theme::dim(),
        ),
        Span::styled("  │  ", Theme::muted()),
        Span::styled(
            format!("optimizer {}", state.optimizer_label),
            Theme::dim(),
        ),
    ]);

    f.render_widget(
        Paragraph::new(line).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Theme::border()),
        ),
        area,
    );
}

fn draw_chart(f: &mut Frame, area: Rect, state: &TrainingState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border())
        .title(" Loss ")
        .title_style(Theme::title());

    if state.loss_series.iter().all(|s| s.is_empty()) {
        f.render_widget(
            Paragraph::new(Span::styled("waiting for data...", Theme::muted()))
                .block(block)
                .alignment(Alignment::Center),
            area,
        );
        return;
    }

    // Compute axis bounds
    let max_epoch = state
        .loss_series
        .iter()
        .flat_map(|s| s.iter().map(|(x, _)| *x))
        .fold(1.0_f64, f64::max);

    let max_loss = state
        .loss_series
        .iter()
        .flat_map(|s| s.iter().map(|(_, y)| *y))
        .fold(0.01_f64, f64::max);

    let datasets: Vec<Dataset> = state
        .loss_series
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_empty())
        .map(|(i, series)| {
            let color = WORKER_COLORS[i % WORKER_COLORS.len()];
            Dataset::default()
                .name(format!("w{i}"))
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(color))
                .data(series)
        })
        .collect();

    let chart = Chart::new(datasets)
        .block(block)
        .x_axis(
            Axis::default()
                .title("epoch")
                .style(Theme::dim())
                .bounds([0.0, max_epoch])
                .labels(vec![
                    Span::styled("0", Theme::muted()),
                    Span::styled(format!("{}", max_epoch as usize), Theme::muted()),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("loss")
                .style(Theme::dim())
                .bounds([0.0, max_loss * 1.1])
                .labels(vec![
                    Span::styled("0", Theme::muted()),
                    Span::styled(format!("{max_loss:.2}"), Theme::muted()),
                ]),
        );

    f.render_widget(chart, area);
}

fn draw_workers_table(f: &mut Frame, area: Rect, state: &TrainingState) {
    let header = Row::new(vec![
        Cell::from("id").style(Theme::title()),
        Cell::from("epochs").style(Theme::title()),
        Cell::from("last loss").style(Theme::title()),
        Cell::from("status").style(Theme::title()),
    ]);

    let rows: Vec<Row> = state
        .workers
        .iter()
        .map(|w| {
            let status_cell = if w.done {
                Cell::from("done").style(Theme::muted())
            } else {
                Cell::from("active").style(Theme::ok())
            };

            let loss_str = w
                .last_loss
                .map(|l| format!("{l:.4}"))
                .unwrap_or_else(|| "—".into());

            Row::new(vec![
                Cell::from(format!("{}", w.id)).style(Theme::text()),
                Cell::from(format!("{}/{}", w.epochs_done, state.max_epochs))
                    .style(Theme::text()),
                Cell::from(loss_str).style(Theme::text()),
                status_cell,
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(4),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border())
            .title(" Workers ")
            .title_style(Theme::title()),
    );

    f.render_widget(table, area);
}

fn draw_params(f: &mut Frame, area: Rect, state: &TrainingState) {
    let params = state.final_params.as_ref().unwrap();

    let preview: String = params
        .iter()
        .take(6)
        .map(|p| format!("{p:.4}"))
        .collect::<Vec<_>>()
        .join("  ");

    let suffix = if params.len() > 6 {
        format!("  … +{} more", params.len() - 6)
    } else {
        String::new()
    };

    let line = Line::from(vec![
        Span::styled(preview, Theme::ok()),
        Span::styled(suffix, Theme::muted()),
    ]);

    f.render_widget(
        Paragraph::new(line)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Theme::accent_magenta())
                    .title(" Final Parameters ")
                    .title_style(
                        Theme::accent_magenta().add_modifier(Modifier::BOLD),
                    ),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_log(f: &mut Frame, area: Rect, state: &TrainingState) {
    let log_height = area.height.saturating_sub(2) as usize;
    let tail = state
        .logs
        .iter()
        .rev()
        .take(log_height)
        .rev()
        .map(|(level, msg)| {
            let (tag, tag_style) = match level {
                LogLevel::Info => ("[info] ", Theme::dim()),
                LogLevel::Warn => ("[warn] ", Theme::warn()),
                LogLevel::Error => ("[error]", Theme::error()),
            };
            Line::from(vec![
                Span::styled(tag, tag_style),
                Span::styled(msg.as_str(), Theme::text()),
            ])
        })
        .collect::<Vec<_>>();

    f.render_widget(
        Paragraph::new(tail)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Theme::border())
                    .title(" Events ")
                    .title_style(Theme::title()),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}