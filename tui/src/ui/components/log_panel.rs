use ratatui::{
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::ui::screens::training::{LogLevel, TrainingState};
use crate::ui::theme::Theme;

/// Draws the scrolling event log panel showing the most recent entries.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
pub fn draw_log(f: &mut Frame, area: Rect, state: &TrainingState) {
    let log_height = area.height.saturating_sub(2) as usize;
    let tail = state
        .logs
        .iter()
        .rev()
        .take(log_height)
        .rev()
        .map(|(level, msg)| {
            let (tag, tag_style) = match level {
                LogLevel::Info => ("[info] ", Theme::dim()),
                LogLevel::Warn => ("[warn] ", Theme::warn()),
                LogLevel::Error => ("[error]", Theme::error()),
            };
            Line::from(vec![
                Span::styled(tag, tag_style),
                Span::styled(msg.as_str(), Theme::text()),
            ])
        })
        .collect::<Vec<_>>();

    f.render_widget(
        Paragraph::new(tail)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Theme::border())
                    .title(" Events ")
                    .title_style(Theme::title()),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}
