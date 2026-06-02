use ratatui::{
    layout::Rect,
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use orchestrator::StopReason;

use crate::ui::screens::training::{Phase, TrainingState, TrainingView};
use crate::ui::theme::Theme;
use crate::ui::utils::fmt_loss;

/// Draws the top header bar with session phase, elapsed time, worker/server counts and optimizer.
///
/// # Args
/// * `f` - The ratatui frame to draw into.
/// * `area` - The area to render into.
/// * `state` - The current training screen state.
pub fn draw_header(f: &mut Frame, area: Rect, state: &TrainingState) {
    let phase_span = match state.phase {
        Phase::Converting => Span::styled("CONVERTING", Theme::accent_cyan()),
        Phase::Connecting => Span::styled("CONNECTING", Theme::accent_cyan()),
        Phase::Training => Span::styled("TRAINING", Theme::ok()),
        Phase::Finished => match state.finish_reason {
            Some(StopReason::EarlyStopping) => {
                Span::styled("FINISHED · early stop", Theme::accent_magenta())
            }
            Some(StopReason::ManualStop) => {
                Span::styled("FINISHED · stopped", Theme::accent_magenta())
            }
            _ => Span::styled("FINISHED", Theme::accent_magenta()),
        },
        Phase::Error => Span::styled("ERROR", Theme::error()),
    };

    let view_hint = match state.view {
        TrainingView::Dashboard => "  [v] topology",
        TrainingView::Topology => "  [v] dashboard",
    };

    let hint = if state.final_trained.is_some() {
        Span::styled(
            format!("  [q] menu  [←/→] worker  [s] save{view_hint}"),
            Theme::muted(),
        )
    } else if state.is_active() {
        Span::styled(
            format!("  [q] leave  [←/→] worker  [x] stop{view_hint}"),
            Theme::muted(),
        )
    } else {
        Span::styled(
            format!("  [q] menu  [←/→] worker{view_hint}"),
            Theme::muted(),
        )
    };

    let workers_done = state.workers_done();

    let mut spans = vec![
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
    ];

    if state.servers_total > 0 {
        spans.push(Span::styled(
            format!("servers {}", state.servers_total),
            Theme::dim(),
        ));
        spans.push(Span::styled("  │  ", Theme::muted()));
    }

    spans.push(Span::styled(
        format!("avg loss {}", avg_loss_str(state)),
        Theme::dim(),
    ));
    spans.push(Span::styled("  │  ", Theme::muted()));
    spans.push(Span::styled(
        format!("optimizer {}", state.optimizer_label),
        Theme::dim(),
    ));

    if let Some(tol) = &state.early_stopping_label {
        spans.push(Span::styled("  │  ", Theme::muted()));
        spans.push(Span::styled(format!("early stop tol {tol}"), Theme::dim()));
    }

    spans.push(hint);

    let line = Line::from(spans);

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
    let count = state.workers.len();

    if count == 0 {
        return "—".into();
    }

    let sum: f64 = state
        .workers
        .iter()
        .flat_map(|worker| worker.last_loss)
        .sum();

    fmt_loss(sum / count as f64)
}
