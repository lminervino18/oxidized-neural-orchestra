use ratatui::{
    layout::Rect,
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::ui::screens::training::TrainingState;
use crate::ui::theme::Theme;

/// Draws the final parameters preview panel shown after training completes.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
pub fn draw_params(f: &mut Frame, area: Rect, state: &TrainingState) {
    let params = state.final_params.as_ref().unwrap();

    let preview: String = params
        .iter()
        .take(6)
        .map(|p| format!("{p:.4}"))
        .collect::<Vec<_>>()
        .join("  ");

    let suffix = if params.len() > 6 {
        format!("  … +{} more", params.len() - 6)
    } else {
        String::new()
    };

    let line = Line::from(vec![
        Span::styled(preview, Theme::ok()),
        Span::styled(suffix, Theme::muted()),
    ]);

    f.render_widget(
        Paragraph::new(line)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Theme::accent_magenta())
                    .title(" Final Parameters ")
                    .title_style(Theme::accent_magenta().add_modifier(Modifier::BOLD)),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}