use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::ui::theme::Theme;

use super::{Action, Screen};

const LOGO: &str = r#"
  ██████╗ ███╗   ██╗ ██████╗ 
 ██╔═══██╗████╗  ██║██╔═══██╗
 ██║   ██║██╔██╗ ██║██║   ██║
 ██║   ██║██║╚██╗██║██║   ██║
 ╚██████╔╝██║ ╚████║╚██████╔╝
  ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ 
    
oxidized-neural-orchestra
"#;

const MENU_ITEMS: &[&str] = &["Start Training", "Quit"];

pub struct MenuState {
    pub selected: usize,
}

impl MenuState {
    pub fn new() -> Self {
        Self { selected: 0 }
    }
}

pub fn handle_key(state: &mut MenuState, key: KeyCode) -> Action {
    match key {
        KeyCode::Up | KeyCode::Char('k') => {
            if state.selected > 0 {
                state.selected -= 1;
            }
            Action::None
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if state.selected < MENU_ITEMS.len() - 1 {
                state.selected += 1;
            }
            Action::None
        }
        KeyCode::Enter => match state.selected {
            0 => Action::Transition(Screen::Config(super::config::ConfigState::new())),
            1 => Action::Quit,
            _ => Action::None,
        },
        KeyCode::Char('q') => Action::Quit,
        _ => Action::None,
    }
}

pub fn draw(f: &mut Frame, state: &MenuState) {
    let area = f.size();
    f.render_widget(Block::default().style(Theme::base()), area);

    let outer = centered_rect(60, 70, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),
            Constraint::Length(1),
            Constraint::Length(MENU_ITEMS.len() as u16 * 2 + 2),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(outer);

    draw_logo(f, chunks[0]);
    draw_menu(f, chunks[2], state);
    draw_hint(f, chunks[4]);
}

fn draw_logo(f: &mut Frame, area: Rect) {
    let lines: Vec<Line> = LOGO
        .lines()
        .map(|l| Line::from(Span::styled(l, Theme::title())))
        .collect();

    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), area);
}

fn draw_menu(f: &mut Frame, area: Rect, state: &MenuState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border())
        .title(" MENU ")
        .title_alignment(Alignment::Center)
        .title_style(Theme::title());

    let inner = block.inner(area);
    f.render_widget(block, area);

    let item_areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            MENU_ITEMS
                .iter()
                .map(|_| Constraint::Length(2))
                .collect::<Vec<_>>(),
        )
        .split(inner);

    for (i, (label, item_area)) in MENU_ITEMS.iter().zip(item_areas.iter()).enumerate() {
        let is_selected = i == state.selected;
        let (prefix, style) = if is_selected {
            ("▶ ", Theme::title().add_modifier(Modifier::BOLD))
        } else {
            ("  ", Theme::dim())
        };

        let line = Line::from(vec![
            Span::styled(prefix, style),
            Span::styled(*label, style),
        ]);

        f.render_widget(Paragraph::new(line).wrap(Wrap { trim: true }), *item_area);
    }
}

fn draw_hint(f: &mut Frame, area: Rect) {
    let hint = Paragraph::new(Line::from(vec![
        Span::styled("↑↓ / j k", Theme::dim()),
        Span::styled("  navigate    ", Theme::muted()),
        Span::styled("enter", Theme::dim()),
        Span::styled("  select    ", Theme::muted()),
        Span::styled("q", Theme::dim()),
        Span::styled("  quit", Theme::muted()),
    ]))
    .alignment(Alignment::Center);

    f.render_widget(hint, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
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
