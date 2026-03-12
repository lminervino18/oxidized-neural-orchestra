use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Returns a centered sub-rectangle of `r` with the given percentage dimensions.
///
/// # Args
/// * `percent_x` - Horizontal size as a percentage of `r`.
/// * `percent_y` - Vertical size as a percentage of `r`.
/// * `r` - The outer rectangle to center within.
///
/// # Returns
/// A `Rect` centered inside `r`.
pub fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vert[1])[1]
}

/// Formats a loss value with adaptive precision.
///
/// Uses scientific notation for values smaller than `1e-4` to avoid
/// displaying them as `0.00000000` at fixed precision.
///
/// # Args
/// * `loss` - The loss value to format.
///
/// # Returns
/// A formatted string representation of the loss.
pub fn fmt_loss(loss: f32) -> String {
    if loss.abs() < 1e-4 {
        format!("{loss:.3e}")
    } else {
        format!("{loss:.8}")
    }
}