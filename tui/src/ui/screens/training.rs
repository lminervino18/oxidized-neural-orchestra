use std::time::Instant;

use crossterm::event::KeyCode;
use orchestrator::{
    configs::{ModelConfig, TrainingConfig},
    TrainingEvent,
};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table, Wrap},
    Frame,
};
use tokio::sync::mpsc;

use crate::ui::theme::Theme;

use super::Action;

const WORKER_COLORS: &[Color] = &[
    Color::Rgb(57, 255, 20),
    Color::Rgb(0, 255, 255),
    Color::Rgb(255, 0, 255),
    Color::Rgb(255, 255, 0),
    Color::Rgb(255, 130, 0),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Connecting,
    Training,
    Finished,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfirmQuit {
    Hidden,
    Visible,
}

#[derive(Debug, Clone)]
struct WorkerState {
    id: usize,
    epochs_done: usize,
    last_loss: Option<f32>,
    done: bool,
}

pub struct TrainingState {
    workers_total: usize,
    optimizer_label: String,
    max_epochs: usize,
    phase: Phase,
    started_at: Instant,
    workers: Vec<WorkerState>,
    loss_series: Vec<Vec<(f64, f64)>>,
    logs: Vec<(LogLevel, String)>,
    final_params: Option<Vec<f32>>,
    error: Option<String>,
    events: mpsc::Receiver<TrainingEvent>,
    confirm_quit: ConfirmQuit,
    /// Index of the worker currently shown in the per-worker chart.
    selected_worker: usize,
}

#[derive(Debug, Clone, Copy)]
enum LogLevel {
    Info,
    Warn,
    Error,
}

impl TrainingState {
    pub fn new(model: ModelConfig, training: TrainingConfig<String>, workers_total: usize) -> Self {
        let optimizer_label = format!("{:?}", training.optimizer)
            .split_whitespace()
            .next()
            .unwrap_or("unknown")
            .to_string();

        let max_epochs = training.max_epochs.get();

        let mut session = match orchestrator::train(model, training) {
            Ok(s) => s,
            Err(e) => {
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

        Self {
            workers_total,
            optimizer_label,
            max_epochs,
            phase: Phase::Connecting,
            started_at: Instant::now(),
            workers,
            loss_series,
            logs: vec![(
                LogLevel::Info,
                "connecting to workers and parameter server...".into(),
            )],
            final_params: None,
            error: None,
            events,
            confirm_quit: ConfirmQuit::Hidden,
            selected_worker: 0,
        }
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
            confirm_quit: ConfirmQuit::Hidden,
            selected_worker: 0,
        }
    }

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

    fn is_active(&self) -> bool {
        matches!(self.phase, Phase::Connecting | Phase::Training)
    }

    /// Builds the average loss series across all workers.
    fn avg_loss_series(&self) -> Vec<(f64, f64)> {
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

    fn next_worker(&mut self) {
        if self.workers_total > 0 {
            self.selected_worker = (self.selected_worker + 1) % self.workers_total;
        }
    }

    fn prev_worker(&mut self) {
        if self.workers_total > 0 {
            self.selected_worker =
                (self.selected_worker + self.workers_total - 1) % self.workers_total;
        }
    }
}

pub fn handle_key(state: &mut TrainingState, key: KeyCode) -> Action {
    match (key, state.confirm_quit, state.is_active()) {
        // Navigate workers with arrows (always available)
        (KeyCode::Left, ConfirmQuit::Hidden, _) => {
            state.prev_worker();
            Action::None
        }
        (KeyCode::Right, ConfirmQuit::Hidden, _) => {
            state.next_worker();
            Action::None
        }
        // During training: q/esc opens confirmation
        (KeyCode::Char('q') | KeyCode::Esc, ConfirmQuit::Hidden, true) => {
            state.confirm_quit = ConfirmQuit::Visible;
            Action::None
        }
        // Confirmation: y confirms exit
        (KeyCode::Char('y') | KeyCode::Char('Y'), ConfirmQuit::Visible, _) => {
            Action::Transition(super::Screen::Menu(
                crate::ui::screens::menu::MenuState::new(),
            ))
        }
        // Confirmation: n or esc cancels
        (KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc, ConfirmQuit::Visible, _) => {
            state.confirm_quit = ConfirmQuit::Hidden;
            Action::None
        }
        // Finished or error: q/esc goes back to menu
        (KeyCode::Char('q') | KeyCode::Esc, ConfirmQuit::Hidden, false) => {
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

fn draw_charts(f: &mut Frame, area: Rect, state: &TrainingState) {
    let halves = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
        .split(area);

    draw_avg_chart(f, halves[0], state);
    draw_selected_worker_chart(f, halves[1], state);
}

fn draw_avg_chart(f: &mut Frame, area: Rect, state: &TrainingState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border())
        .title(" Average Loss — all workers ")
        .title_style(Theme::title());

    let avg = state.avg_loss_series();

    if avg.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("waiting for data...", Theme::muted()))
                .block(block)
                .alignment(Alignment::Center),
            area,
        );
        return;
    }

    let max_epoch = avg.iter().map(|(x, _)| *x).fold(1.0_f64, f64::max);
    let max_loss = avg.iter().map(|(_, y)| *y).fold(0.01_f64, f64::max);

    let dataset = Dataset::default()
        .name("avg")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::White))
        .data(&avg);

    let chart = Chart::new(vec![dataset])
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

fn draw_selected_worker_chart(f: &mut Frame, area: Rect, state: &TrainingState) {
    let worker_id = state.selected_worker;
    let color = WORKER_COLORS[worker_id % WORKER_COLORS.len()];

    let title = format!(
        " Worker {} — Loss  [←/→ to switch] ",
        worker_id
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .title(title)
        .title_style(Style::default().fg(color).add_modifier(Modifier::BOLD));

    let series = match state.loss_series.get(worker_id) {
        Some(s) => s,
        None => {
            f.render_widget(
                Paragraph::new(Span::styled("no data", Theme::muted()))
                    .block(block)
                    .alignment(Alignment::Center),
                area,
            );
            return;
        }
    };

    if series.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("waiting for data...", Theme::muted()))
                .block(block)
                .alignment(Alignment::Center),
            area,
        );
        return;
    }

    let max_epoch = series.iter().map(|(x, _)| *x).fold(1.0_f64, f64::max);
    let max_loss = series.iter().map(|(_, y)| *y).fold(0.01_f64, f64::max);

    let dataset = Dataset::default()
        .name(format!("w{worker_id}"))
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color))
        .data(series);

    let chart = Chart::new(vec![dataset])
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

fn draw_confirm_quit(f: &mut Frame, area: Rect) {
    let popup = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(45),
            Constraint::Length(5),
            Constraint::Percentage(55),
        ])
        .split(area)[1];

    let popup = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(35),
            Constraint::Length(36),
            Constraint::Percentage(35),
        ])
        .split(popup)[1];

    let text = vec![
        Line::from(Span::styled("Training is still running.", Theme::warn())),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled("[y]", Theme::ok()),
            Span::styled(" back to menu    ", Theme::text()),
            Span::styled("[n]", Theme::error()),
            Span::styled(" keep training", Theme::text()),
        ]),
    ];

    f.render_widget(
        Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Theme::warn())
                    .title(" Leave? ")
                    .title_style(Theme::warn().add_modifier(Modifier::BOLD)),
            )
            .alignment(Alignment::Center),
        popup,
    );
}

fn draw_header(f: &mut Frame, area: Rect, state: &TrainingState) {
    let phase_span = match state.phase {
        Phase::Connecting => Span::styled("CONNECTING", Theme::accent_cyan()),
        Phase::Training => Span::styled("TRAINING", Theme::ok()),
        Phase::Finished => Span::styled("FINISHED", Theme::accent_magenta()),
        Phase::Error => Span::styled("ERROR", Theme::error()),
    };

    let hint = if state.is_active() {
        Span::styled("  [q] leave  [←/→] worker", Theme::muted())
    } else {
        Span::styled("  [q] menu  [←/→] worker", Theme::muted())
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
        Span::styled(format!("optimizer {}", state.optimizer_label), Theme::dim()),
        hint,
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
            let is_selected = w.id == state.selected_worker;

            let status_cell = if w.done {
                Cell::from("done").style(Theme::muted())
            } else {
                Cell::from("active").style(Theme::ok())
            };

            let id_style = if is_selected {
                let color = WORKER_COLORS[w.id % WORKER_COLORS.len()];
                Style::default().fg(color).add_modifier(Modifier::BOLD)
            } else {
                Theme::text()
            };

            let loss_str = w
                .last_loss
                .map(|l| format!("{l:.4}"))
                .unwrap_or_else(|| "—".into());

            Row::new(vec![
                Cell::from(format!("{}", w.id)).style(id_style),
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
                    .title_style(Theme::accent_magenta().add_modifier(Modifier::BOLD)),
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