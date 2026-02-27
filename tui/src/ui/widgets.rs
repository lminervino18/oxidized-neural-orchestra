// use ratatui::{
//     layout::Constraint,
//     style::{Modifier, Style},
//     text::{Line, Span},
//     widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
// };

// use crate::state::model::{SessionPhase, SessionView, WorkerStatus};
// use crate::ui::theme::Theme;

// pub fn header<'a>(view: &'a SessionView) -> Paragraph<'a> {
//     let phase = match view.phase {
//         SessionPhase::Init => Span::styled("INIT", Theme::dim()),
//         SessionPhase::Connecting => Span::styled("CONNECTING", Theme::accent_cyan()),
//         SessionPhase::Training => Span::styled("TRAINING", Theme::ok()),
//         SessionPhase::Finished => Span::styled("FINISHED", Theme::accent_magenta()),
//         SessionPhase::Error => Span::styled("ERROR", Theme::error()),
//     };

//     let line1 = Line::from(vec![
//         Span::styled(
//             "Oxidized Neural Orchestra",
//             Theme::title(),
//         ),
//         Span::raw("  |  "),
//         Span::raw("Session: "),
//         phase,
//     ]);

//     let line2 = Line::from(vec![Span::styled(
//         format!(
//             "Elapsed: {:02}:{:02}  |  Steps: {} / {}  |  Workers: {} / {}",
//             view.elapsed.as_secs() / 60,
//             view.elapsed.as_secs() % 60,
//             view.step_done,
//             view.step_total,
//             view.workers_connected,
//             view.workers_total
//         ),
//         Theme::text(),
//     )]);

//     Paragraph::new(vec![line1, line2])
//         .block(panel_block("Overview"))
//         .style(Theme::text())
//         .wrap(Wrap { trim: true })
// }

// pub fn diagram<'a>(view: &'a SessionView) -> Paragraph<'a> {
//     let mut lines: Vec<Line> = Vec::new();

//     lines.push(Line::from(vec![Span::styled("Orchestrator", Theme::title())]));
//     lines.push(Line::from(vec![Span::styled("   |", Theme::dim())]));
//     lines.push(Line::from(vec![
//         Span::styled("ParameterServer ", Theme::text()),
//         Span::styled(format!("({})", view.server.trainer_kind), Theme::accent_cyan()),
//     ]));
//     lines.push(Line::from(vec![Span::styled("   |", Theme::dim())]));
//     lines.push(Line::from(vec![
//         Span::styled("Workers: ", Theme::text()),
//         Span::styled(
//             format!("{} connected", view.workers_connected),
//             Theme::ok(),
//         ),
//     ]));
//     lines.push(Line::from(""));

//     // ASCII row with per-worker status coloring.
//     let mut row: Vec<Span> = Vec::new();
//     for w in view.workers.iter() {
//         let s = status_short(w.status);
//         let st = status_style(w.status);
//         row.push(Span::styled(format!("[W{}:{}] ", w.worker_id, s), st));
//     }
//     lines.push(Line::from(row));

//     Paragraph::new(lines)
//         .block(panel_block("Architecture"))
//         .style(Theme::text())
//         .wrap(Wrap { trim: true })
// }

// pub fn server<'a>(view: &'a SessionView) -> Paragraph<'a> {
//     let lines = vec![
//         Line::from(vec![
//             Span::styled("trainer: ", Theme::dim()),
//             Span::styled(view.server.trainer_kind, Theme::accent_cyan()),
//         ]),
//         Line::from(vec![
//             Span::styled("optimizer: ", Theme::dim()),
//             Span::styled(view.server.optimizer_kind, Theme::text()),
//         ]),
//         Line::from(vec![
//             Span::styled("shard_size: ", Theme::dim()),
//             Span::styled(view.server.shard_size.to_string(), Theme::text()),
//         ]),
//         Line::from(vec![
//             Span::styled("num_params: ", Theme::dim()),
//             Span::styled(view.server.num_params.to_string(), Theme::text()),
//         ]),
//     ];

//     Paragraph::new(lines)
//         .block(panel_block("Parameter Server"))
//         .style(Theme::text())
//         .wrap(Wrap { trim: true })
// }

// pub fn workers_table<'a>(view: &'a SessionView) -> Table<'a> {
//     let header = Row::new(vec!["id", "step", "status", "strategy"])
//         .style(Theme::title());

//     let rows = view.workers.iter().map(|w| {
//         let st = status_style(w.status);
//         let status_cell = Cell::from(status_long(w.status)).style(st);

//         let row_style = match w.status {
//             WorkerStatus::Computing | WorkerStatus::SendingGradients => Theme::highlight_bg(),
//             WorkerStatus::Disconnected => Theme::muted(),
//             WorkerStatus::Error => Theme::error(),
//             _ => Theme::text(),
//         };

//         Row::new(vec![
//             Cell::from(w.worker_id.to_string()).style(Theme::text()),
//             Cell::from(format!("{}/{}", w.step, w.steps_total)).style(Theme::text()),
//             status_cell,
//             Cell::from(w.strategy_kind).style(Theme::dim()),
//         ])
//         .style(row_style)
//     });

//     Table::new(
//         rows,
//         [
//             Constraint::Length(6),
//             Constraint::Length(12),
//             Constraint::Length(18),
//             Constraint::Min(8),
//         ],
//     )
//     .header(header)
//     .block(panel_block("Workers"))
//     .style(Theme::text())
// }

// pub fn logs<'a>(view: &'a SessionView) -> Paragraph<'a> {
//     let tail = view.logs.iter().rev().take(8).rev();

//     let lines = tail
//         .map(|l| {
//             let level_style = match l.level {
//                 "ERROR" => Theme::error(),
//                 "WARN" => Theme::warn(),
//                 "INFO" => Theme::info(),
//                 _ => Theme::dim(),
//             };

//             Line::from(vec![
//                 Span::styled(format!("[{}] ", l.level), level_style),
//                 Span::styled(l.message.as_str(), Theme::text()),
//             ])
//         })
//         .collect::<Vec<_>>();

//     Paragraph::new(lines)
//         .block(panel_block("Events"))
//         .style(Theme::text())
//         .wrap(Wrap { trim: true })
// }

// fn panel_block(title: &'static str) -> Block<'static> {
//     Block::default()
//         .borders(Borders::ALL)
//         .border_style(Theme::border())
//         .title(title)
//         .title_style(Theme::title())
// }

// fn status_short(s: WorkerStatus) -> &'static str {
//     match s {
//         WorkerStatus::WaitingWeights => "wait",
//         WorkerStatus::Computing => "cpu",
//         WorkerStatus::SendingGradients => "tx",
//         WorkerStatus::Disconnected => "disc",
//         WorkerStatus::Error => "err",
//     }
// }

// fn status_long(s: WorkerStatus) -> &'static str {
//     match s {
//         WorkerStatus::WaitingWeights => "waiting_weights",
//         WorkerStatus::Computing => "computing",
//         WorkerStatus::SendingGradients => "sending_gradients",
//         WorkerStatus::Disconnected => "disconnected",
//         WorkerStatus::Error => "error",
//     }
// }

// fn status_style(s: WorkerStatus) -> Style {
//     match s {
//         WorkerStatus::WaitingWeights => Theme::dim(),
//         WorkerStatus::Computing => Theme::accent_cyan(),
//         WorkerStatus::SendingGradients => Theme::accent_magenta(),
//         WorkerStatus::Disconnected => Theme::muted(),
//         WorkerStatus::Error => Theme::error(),
//     }
// }
