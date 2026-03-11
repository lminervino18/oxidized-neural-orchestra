use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::Span,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
    Frame,
};

use crate::ui::screens::training::{TrainingState, WORKER_COLORS};
use crate::ui::theme::Theme;

/// Formats a loss value for chart axis labels with adaptive precision.
fn fmt_axis_loss(loss: f64) -> String {
    if loss.abs() < 1e-4 {
        format!("{loss:.2e}")
    } else {
        format!("{loss:.4}")
    }
}

/// Draws the average loss chart and the selected worker loss chart stacked vertically.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
pub fn draw_charts(f: &mut Frame, area: Rect, state: &TrainingState) {
    let halves = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
        .split(area);

    draw_avg_chart(f, halves[0], state);
    draw_selected_worker_chart(f, halves[1], state);
}

/// Draws the average loss chart across all workers.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
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
                    Span::styled(fmt_axis_loss(max_loss), Theme::muted()),
                ]),
        );

    f.render_widget(chart, area);
}

/// Draws the loss chart for the currently selected worker.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
fn draw_selected_worker_chart(f: &mut Frame, area: Rect, state: &TrainingState) {
    let worker_id = state.selected_worker;
    let color = WORKER_COLORS[worker_id % WORKER_COLORS.len()];

    let title = format!(" Worker {} — Loss  [←/→ to switch] ", worker_id);

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
                    Span::styled(fmt_axis_loss(max_loss), Theme::muted()),
                ]),
        );

    f.render_widget(chart, area);
}