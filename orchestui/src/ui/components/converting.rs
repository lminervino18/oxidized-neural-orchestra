use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::ui::screens::training::TrainingState;
use crate::ui::theme::Theme;
use crate::ui::utils::centered_rect;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Draws a centered popup shown during `Phase::Converting`.
///
/// Overlays a bordered popup over whatever is behind it, showing a spinner,
/// elapsed time, and a brief explanation so the user understands that a large
/// dataset file is being converted to binary format before training begins.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The full terminal area.
/// * `state` - The current training screen state.
pub fn draw_converting(f: &mut Frame, area: Rect, state: &TrainingState) {
    let popup = centered_rect(55, 40, area);
    f.render_widget(Clear, popup);

    let frame_idx = (state.started_at.elapsed().as_millis() / 80) as usize;
    let spin = SPINNER[frame_idx % SPINNER.len()];
    let elapsed = state.elapsed_str();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::accent_cyan())
        .title(" Preparing Dataset ")
        .title_style(Theme::accent_cyan().add_modifier(Modifier::BOLD));

    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(format!(" {spin}  "), Theme::accent_cyan()),
            Span::styled(
                "converting dataset to binary format...",
                Theme::ok().add_modifier(Modifier::BOLD),
            ),
        ])),
        chunks[1],
    );

    f.render_widget(
        Paragraph::new(Span::styled(
            " This may take a while for large files.",
            Theme::muted(),
        )),
        chunks[2],
    );

    f.render_widget(
        Paragraph::new(Span::styled(
            " The converted file will be cached next to the source.",
            Theme::muted(),
        )),
        chunks[3],
    );

    f.render_widget(
        Paragraph::new(Span::styled(format!(" elapsed  {elapsed}"), Theme::dim())),
        chunks[4],
    );

    f.render_widget(
        Paragraph::new(Span::styled("[q]  cancel", Theme::muted())).alignment(Alignment::Center),
        chunks[6],
    );
}
