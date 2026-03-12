use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::ui::{theme::Theme, utils::centered_rect};

/// Draws a popup prompting the user to enter a file path for saving the model.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `path` - The current input string.
pub fn draw_save_popup(f: &mut Frame, path: &str) {
    let area = centered_rect(50, 30, f.size());
    f.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::accent_magenta())
        .title(" Save Model ")
        .title_style(Theme::accent_magenta().add_modifier(Modifier::BOLD));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Span::styled(
            "Enter output path for the .safetensors file.",
            Theme::muted(),
        )),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(Span::styled(
            "Leave empty to save as model.safetensors",
            Theme::dim(),
        )),
        chunks[1],
    );

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border())
        .title(" path ")
        .title_style(Theme::title());

    let input_inner = input_block.inner(chunks[3]);
    f.render_widget(input_block, chunks[3]);

    let display = if path.is_empty() {
        Line::from(vec![
            Span::styled("model.safetensors", Theme::muted()),
            Span::styled("█", Theme::accent_cyan()),
        ])
    } else {
        Line::from(vec![
            Span::styled(path, Theme::ok()),
            Span::styled("█", Theme::accent_cyan()),
        ])
    };

    f.render_widget(Paragraph::new(display), input_inner);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("[enter]", Theme::accent_cyan()),
            Span::styled(" save  ", Theme::dim()),
            Span::styled("[esc]", Theme::accent_cyan()),
            Span::styled(" cancel", Theme::dim()),
        ]))
        .alignment(Alignment::Center),
        chunks[4],
    );
}
