use ratatui::{
    layout::Constraint,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
};

use crate::state::model::{SessionPhase, SessionView, WorkerStatus};

pub fn header<'a>(view: &'a SessionView) -> Paragraph<'a> {
    let phase = match view.phase {
        SessionPhase::Init => "INIT",
        SessionPhase::Connecting => "CONNECTING",
        SessionPhase::Training => "TRAINING",
        SessionPhase::Finished => "FINISHED",
        SessionPhase::Error => "ERROR",
    };

    let line1 = Line::from(vec![
        Span::styled(
            "Distributed ML Orchestrator",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw("  |  "),
        Span::raw(format!("Session: {phase}")),
    ]);

    let line2 = Line::from(vec![Span::raw(format!(
        "Elapsed: {:02}:{:02}  |  Steps: {} / {}  |  Workers: {} / {}",
        view.elapsed.as_secs() / 60,
        view.elapsed.as_secs() % 60,
        view.step_done,
        view.step_total,
        view.workers_connected,
        view.workers_total
    ))]);

    Paragraph::new(vec![line1, line2])
        .block(Block::default().borders(Borders::ALL).title("Overview"))
        .wrap(Wrap { trim: true })
}

pub fn diagram<'a>(view: &'a SessionView) -> Paragraph<'a> {
    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from("Orchestrator"));
    lines.push(Line::from("   |"));
    lines.push(Line::from(format!(
        "ParameterServer ({})",
        view.server.trainer_kind
    )));
    lines.push(Line::from("   |"));
    lines.push(Line::from(format!(
        "Workers: {} connected",
        view.workers_connected
    )));
    lines.push(Line::from(""));

    let w_row = view
        .workers
        .iter()
        .map(|w| format!("[W{}:{}]", w.worker_id, status_short(w.status)))
        .collect::<Vec<_>>()
        .join(" ");

    lines.push(Line::from(w_row));

    Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Architecture"))
        .wrap(Wrap { trim: true })
}

pub fn server<'a>(view: &'a SessionView) -> Paragraph<'a> {
    let lines = vec![
        Line::from(format!("trainer: {}", view.server.trainer_kind)),
        Line::from(format!("optimizer: {}", view.server.optimizer_kind)),
        Line::from(format!("shard_size: {}", view.server.shard_size)),
        Line::from(format!("num_params: {}", view.server.num_params)),
    ];

    Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Parameter Server"))
        .wrap(Wrap { trim: true })
}

pub fn workers_table<'a>(view: &'a SessionView) -> Table<'a> {
    let header = Row::new(vec!["id", "step", "status", "strategy"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let rows = view.workers.iter().map(|w| {
        Row::new(vec![
            Cell::from(w.worker_id.to_string()),
            Cell::from(format!("{}/{}", w.step, w.steps_total)),
            Cell::from(status_long(w.status)),
            Cell::from(w.strategy_kind),
        ])
    });

    Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Length(12),
            Constraint::Length(18),
            Constraint::Min(8),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Workers"))
}

pub fn logs<'a>(view: &'a SessionView) -> Paragraph<'a> {
    let tail = view.logs.iter().rev().take(8).rev();

    let lines = tail
        .map(|l| {
            Line::from(vec![
                Span::raw(format!("[{}] ", l.level)),
                Span::raw(l.message.as_str()),
            ])
        })
        .collect::<Vec<_>>();

    Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Events"))
        .wrap(Wrap { trim: true })
}

fn status_short(s: WorkerStatus) -> &'static str {
    match s {
        WorkerStatus::WaitingWeights => "wait",
        WorkerStatus::Computing => "cpu",
        WorkerStatus::SendingGradients => "tx",
        WorkerStatus::Disconnected => "disc",
        WorkerStatus::Error => "err",
    }
}

fn status_long(s: WorkerStatus) -> &'static str {
    match s {
        WorkerStatus::WaitingWeights => "waiting_weights",
        WorkerStatus::Computing => "computing",
        WorkerStatus::SendingGradients => "sending_gradients",
        WorkerStatus::Disconnected => "disconnected",
        WorkerStatus::Error => "error",
    }
}
