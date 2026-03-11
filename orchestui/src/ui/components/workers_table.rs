use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

use crate::ui::screens::training::{TrainingState, WORKER_COLORS};
use crate::ui::theme::Theme;

const BAR_WIDTH: usize = 30;

/// Formats a loss value with adaptive precision.
///
/// Uses scientific notation for values smaller than `1e-4` to avoid
/// displaying them as `0.00000000` at fixed precision.
fn fmt_loss(loss: f32) -> String {
    if loss.abs() < 1e-4 {
        format!("{loss:.3e}")
    } else {
        format!("{loss:.8}")
    }
}

/// Draws the global training progress bar followed by the workers status table.
///
/// Progress is based on the maximum epoch count reached across all workers,
/// which best represents overall training advancement regardless of stragglers.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
pub fn draw_workers_table(f: &mut Frame, area: Rect, state: &TrainingState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    draw_progress_bar(f, chunks[0], state);
    draw_table(f, chunks[1], state);
}

/// Draws the global epoch progress bar.
fn draw_progress_bar(f: &mut Frame, area: Rect, state: &TrainingState) {
    let current = state
        .workers
        .iter()
        .map(|w| w.epochs_done)
        .max()
        .unwrap_or(0);

    let max = state.max_epochs;
    let filled = if max > 0 {
        (current * BAR_WIDTH / max).min(BAR_WIDTH)
    } else {
        0
    };
    let empty = BAR_WIDTH - filled;

    let pct = if max > 0 {
        current * 100 / max
    } else {
        0
    };

    let bar_style = if state.phase == crate::ui::screens::training::Phase::Finished {
        Theme::accent_magenta()
    } else {
        Theme::ok()
    };

    let line = Line::from(vec![
        Span::styled("█".repeat(filled), bar_style),
        Span::styled("░".repeat(empty), Theme::muted()),
        Span::styled(
            format!("  {current}/{max}  ({pct}%)"),
            Theme::dim(),
        ),
    ]);

    f.render_widget(
        Paragraph::new(line).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Theme::border())
                .title(" Progress ")
                .title_style(Theme::title()),
        ),
        area,
    );
}

/// Draws the workers status table.
fn draw_table(f: &mut Frame, area: Rect, state: &TrainingState) {
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
                .map(|l| fmt_loss(l))
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
            Constraint::Length(14),
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