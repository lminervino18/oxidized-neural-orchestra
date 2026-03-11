use ratatui::{
    layout::Rect,
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::ui::screens::training::{Phase, TrainingState};
use crate::ui::theme::Theme;

/// Formats a loss value with adaptive precision.
///
/// Uses scientific notation for values smaller than `1e-4` to avoid
/// displaying them as `0.00000000` at fixed precision.
fn fmt_loss(loss: f32) -> String {
    if loss.abs() < 1e-4 {
        format!("{loss:.3e}")
    } else {
        format!("{loss:.8}")
    }
}

/// Draws the top header bar with session phase, elapsed time, worker/server counts and optimizer.
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

    let hint = if state.final_trained.is_some() {
        Span::styled("  [q] menu  [←/→] worker  [s] save", Theme::muted())
    } else if state.is_active() {
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
        Span::styled(format!("servers {}", state.servers_total), Theme::dim()),
        Span::styled("  │  ", Theme::muted()),
        Span::styled(format!("avg loss {}", avg_loss_str(state)), Theme::dim()),
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

fn avg_loss_str(state: &TrainingState) -> String {
    let losses: Vec<f32> = state.workers.iter().filter_map(|w| w.last_loss).collect();

    if losses.is_empty() {
        return "—".into();
    }

    let avg = losses.iter().sum::<f32>() / losses.len() as f32;
    fmt_loss(avg)
}