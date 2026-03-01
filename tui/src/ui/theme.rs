use ratatui::style::{Color, Modifier, Style};

/// Neon-green cyber theme.
///
/// Base aesthetic:
/// - neon green foreground
/// - near-black background
/// - subtle accent colors for statuses
pub struct Theme;

impl Theme {
    // Core palette
    pub const BG: Color = Color::Rgb(0, 0, 0);
    pub const FG_NEON: Color = Color::Rgb(57, 255, 20);
    pub const FG_DIM: Color = Color::Rgb(0, 190, 0);
    pub const FG_MUTED: Color = Color::Rgb(80, 90, 80);

    // Accents (chosen to not clash with neon green)
    pub const ACCENT_CYAN: Color = Color::Rgb(0, 255, 255);
    pub const ACCENT_MAGENTA: Color = Color::Rgb(255, 0, 255);
    pub const ACCENT_YELLOW: Color = Color::Rgb(255, 255, 0);
    pub const ACCENT_RED: Color = Color::Rgb(255, 70, 70);

    /// Default full-screen style.
    pub fn base() -> Style {
        Style::default().fg(Self::FG_NEON).bg(Self::BG)
    }

    /// Panel borders.
    pub fn border() -> Style {
        Style::default().fg(Self::FG_NEON).bg(Self::BG)
    }

    /// Titles (bold neon).
    pub fn title() -> Style {
        Style::default()
            .fg(Self::FG_NEON)
            .add_modifier(Modifier::BOLD)
    }

    /// Regular text.
    pub fn text() -> Style {
        Style::default().fg(Self::FG_NEON)
    }

    /// Secondary/dim text.
    pub fn dim() -> Style {
        Style::default().fg(Self::FG_DIM)
    }

    /// Muted/disabled text.
    pub fn muted() -> Style {
        Style::default().fg(Self::FG_MUTED)
    }

    /// Highlight row background.
    pub fn highlight_bg() -> Style {
        Style::default()
            .bg(Color::Rgb(0, 30, 0))
            .add_modifier(Modifier::BOLD)
    }

    pub fn ok() -> Style {
        Style::default()
            .fg(Self::FG_NEON)
            .add_modifier(Modifier::BOLD)
    }

    pub fn warn() -> Style {
        Style::default()
            .fg(Self::ACCENT_YELLOW)
            .add_modifier(Modifier::BOLD)
    }

    pub fn error() -> Style {
        Style::default()
            .fg(Self::ACCENT_RED)
            .add_modifier(Modifier::BOLD)
    }

    pub fn info() -> Style {
        Style::default().fg(Self::FG_DIM)
    }

    pub fn accent_cyan() -> Style {
        Style::default()
            .fg(Self::ACCENT_CYAN)
            .add_modifier(Modifier::BOLD)
    }

    pub fn accent_magenta() -> Style {
        Style::default()
            .fg(Self::ACCENT_MAGENTA)
            .add_modifier(Modifier::BOLD)
    }
}
