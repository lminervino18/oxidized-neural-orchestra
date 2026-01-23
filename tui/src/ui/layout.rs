use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Computes the main layout regions.
///
/// # Returns
/// (header, body, logs_opt)
pub fn vertical(area: Rect, show_logs: bool) -> (Rect, Rect, Option<Rect>) {
    let constraints = if show_logs {
        vec![Constraint::Length(4), Constraint::Min(10), Constraint::Length(10)]
    } else {
        vec![Constraint::Length(4), Constraint::Min(10)]
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let header = chunks[0];
    let body = chunks[1];
    let logs = if show_logs { Some(chunks[2]) } else { None };

    (header, body, logs)
}

/// Splits body into (diagram_opt, right).
pub fn body(area: Rect, show_diagram: bool) -> (Option<Rect>, Rect) {
    if !show_diagram {
        return (None, area);
    }

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([ratatui::layout::Constraint::Percentage(40), ratatui::layout::Constraint::Percentage(60)])
        .split(area);

    (Some(cols[0]), cols[1])
}

/// Splits right into (server, workers).
pub fn right(area: Rect) -> (Rect, Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Min(6)])
        .split(area);

    (rows[0], rows[1])
}
