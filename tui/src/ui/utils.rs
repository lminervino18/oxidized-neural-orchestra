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