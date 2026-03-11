use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::ui::theme::Theme;

/// Draws the quit confirmation popup over the current screen.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The full terminal area used to center the popup.
pub fn draw_confirm_quit(f: &mut Frame, area: Rect) {
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
