use ratatui::{
    layout::Alignment,
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::ui::{theme::Theme, utils::centered_rect_fixed};

/// Draws the stop-confirmation popup centered over the current screen.
pub fn draw_confirm_stop(f: &mut Frame) {
    let area = centered_rect_fixed(50, 5, f.size());
    f.render_widget(Clear, area);
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "Stop training at this epoch boundary?",
                Theme::warn(),
            )),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled("[y]", Theme::ok()),
                Span::styled(" stop training      ", Theme::text()),
                Span::styled("[n]", Theme::error()),
                Span::styled(" keep training", Theme::text()),
            ]),
        ])
        .block(
            Block::default()
                .style(Theme::base())
                .borders(Borders::ALL)
                .border_style(Theme::warn())
                .title(" Stop? ")
                .title_style(Theme::warn().add_modifier(Modifier::BOLD)),
        )
        .alignment(Alignment::Center),
        area,
    );
}
