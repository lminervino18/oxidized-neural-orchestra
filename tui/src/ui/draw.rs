use ratatui::{Frame, layout::Rect};

use crate::state::model::SessionView;

/// Draws the whole UI.
///
/// # Args
/// * `f` - Frame provided by ratatui.
/// * `view` - Snapshot to render.
/// * `show_diagram` - Whether to render the architecture diagram.
/// * `show_logs` - Whether to render the logs panel.
pub fn draw(f: &mut Frame, view: &SessionView, _show_diagram: bool, _show_logs: bool) {
    let area: Rect = f.size();
    // Temporary stub: we'll implement panels in the next commit.
    let _ = (view, area);
}
