use ratatui::{widgets::Block, Frame};

use crate::state::model::SessionView;

use super::{layout, theme::Theme, widgets};

/// Draws the entire UI.
pub fn draw(f: &mut Frame, view: &SessionView, show_diagram: bool, show_logs: bool) {
    let area = f.size();

    // Paint full background for consistent neon theme.
    f.render_widget(Block::default().style(Theme::base()), area);

    let (header_area, body_area, logs_area) = layout::vertical(area, show_logs);
    let (diagram_area, right_area) = layout::body(body_area, show_diagram);
    let (server_area, workers_area) = layout::right(right_area);

    f.render_widget(widgets::header(view), header_area);

    if let Some(diag) = diagram_area {
        f.render_widget(widgets::diagram(view), diag);
    }

    f.render_widget(widgets::server(view), server_area);
    f.render_widget(widgets::workers_table(view), workers_area);

    if let Some(logs) = logs_area {
        f.render_widget(widgets::logs(view), logs);
    }
}
