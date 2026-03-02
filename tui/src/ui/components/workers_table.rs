use ratatui::{
    layout::{Constraint, Rect},
    style::{Modifier, Style},
    widgets::{Block, Borders, Cell, Row, Table},
    Frame,
};

use crate::ui::screens::training::{TrainingState, WORKER_COLORS};
use crate::ui::theme::Theme;

/// Draws the workers status table showing epochs, last loss and connection state.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
pub fn draw_workers_table(f: &mut Frame, area: Rect, state: &TrainingState) {
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
                Cell::from(format!("{}/{}", w.epochs_done, state.max_epochs)).style(Theme::text()),
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
