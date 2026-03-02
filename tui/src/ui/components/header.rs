use ratatui::{
    layout::Rect,
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::ui::screens::training::{Phase, TrainingState};
use crate::ui::theme::Theme;

/// Draws the top header bar with session phase, elapsed time, worker count and optimizer.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
pub fn draw_header(f: &mut Frame, area: Rect, state: &TrainingState) {
    let phase_span = match state.phase {
        Phase::Connecting => Span::styled("CONNECTING", Theme::accent_cyan()),
        Phase::Training => Span::styled("TRAINING", Theme::ok()),
        Phase::Finished => Span::styled("FINISHED", Theme::accent_magenta()),
        Phase::Error => Span::styled("ERROR", Theme::error()),
    };

    let hint = if state.is_active() {
        Span::styled("  [q] leave  [←/→] worker", Theme::muted())
    } else {
        Span::styled("  [q] menu  [←/→] worker", Theme::muted())
    };

    let workers_done = state.workers_done();

    let line = Line::from(vec![
        Span::styled(" ONO  ", Theme::title().add_modifier(Modifier::BOLD)),
        Span::styled("│  ", Theme::muted()),
        phase_span,
        Span::styled("  │  ", Theme::muted()),
        Span::styled(format!("elapsed {}", state.elapsed_str()), Theme::dim()),
        Span::styled("  │  ", Theme::muted()),
        Span::styled(
            format!("workers {}/{}", workers_done, state.workers_total),
            Theme::dim(),
        ),
        Span::styled("  │  ", Theme::muted()),
        Span::styled(format!("optimizer {}", state.optimizer_label), Theme::dim()),
        hint,
    ]);

    f.render_widget(
        Paragraph::new(line).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Theme::border()),
        ),
        area,
    );
}