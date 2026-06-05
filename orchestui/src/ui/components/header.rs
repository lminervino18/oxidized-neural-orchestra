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

    // After a StrategySwitch, some workers became servers — show the real counts.
    let ps_servers = if state.ps_server_count > 0 {
        state.ps_server_count
    } else {
        state.servers_total
    };
    let effective_total = state.workers_total - state.ps_server_count;
    let workers_active = state.workers.iter().filter(|w| !w.done && !w.became_server).count();

    let workers_str = if state.phase == Phase::Finished {
        format!("workers {effective_total}/{effective_total}")
    } else {
        format!("workers {workers_active}/{effective_total}")
    };

    let mut spans = vec![
        Span::styled(" ONO  ", Theme::title().add_modifier(Modifier::BOLD)),
        Span::styled("│  ", Theme::muted()),
        phase_span,
        Span::styled("  │  ", Theme::muted()),
        Span::styled(format!("elapsed {}", state.elapsed_str()), Theme::dim()),
        Span::styled("  │  ", Theme::muted()),
        Span::styled(workers_str, Theme::dim()),
        Span::styled("  │  ", Theme::muted()),
    ];

    if ps_servers > 0 {
        spans.push(Span::styled(format!("servers {ps_servers}"), Theme::dim()));
        spans.push(Span::styled("  │  ", Theme::muted()));
    }

    // Transient badge shown right after a worker starts converting to a server.
    if let Some(wid) = state.converting_worker() {
        spans.push(Span::styled(
            format!("⟳ converting worker {wid} → PS"),
            Theme::accent_magenta().add_modifier(Modifier::BOLD),
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
