use ratatui::{
    layout::Alignment,
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::ui::{theme::Theme, utils::centered_rect_fixed};

/// Draws the quit confirmation popup centered over the current screen.
///
/// `finished` tailors the message: once training is done the model is already
/// saved, so leaving only means returning to the menu.
pub fn draw_confirm_quit(f: &mut Frame, finished: bool) {
    let area = centered_rect_fixed(44, 5, f.size());
    f.render_widget(Clear, area);
    let (headline, stay) = if finished {
        ("Training finished — model saved.", " stay here")
    } else {
        ("Training is still running.", " keep training")
    };
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(headline, Theme::warn())),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled("[y]", Theme::ok()),
                Span::styled(" back to menu      ", Theme::text()),
                Span::styled("[n]", Theme::error()),
                Span::styled(stay, Theme::text()),
            ]),
        ])
        .block(
            Block::default()
                .style(Theme::base())
                .borders(Borders::ALL)
                .border_style(Theme::warn())
                .title(" Leave? ")
                .title_style(Theme::warn().add_modifier(Modifier::BOLD)),
        )
        .alignment(Alignment::Center),
        area,
    );
}
