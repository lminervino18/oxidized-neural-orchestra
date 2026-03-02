use ratatui::style::{Color, Modifier, Style};

/// Centralized theme for the TUI application.
///
/// Defines the full color palette and style helpers used across all screens.
/// All rendering code should use these methods rather than inline colors.
pub struct Theme;

impl Default for Theme {
    fn default() -> Self {
        Self
    }
}

impl Theme {
    /// Near-black background color.
    pub const BG: Color = Color::Rgb(0, 0, 0);
    /// Primary neon-green foreground.
    pub const FG_NEON: Color = Color::Rgb(57, 255, 20);
    /// Dimmed green for secondary text.
    pub const FG_DIM: Color = Color::Rgb(0, 190, 0);
    /// Muted gray-green for disabled or placeholder text.
    pub const FG_MUTED: Color = Color::Rgb(80, 90, 80);
    /// Cyan accent for interactive elements.
    pub const ACCENT_CYAN: Color = Color::Rgb(0, 255, 255);
    /// Magenta accent for completion states.
    pub const ACCENT_MAGENTA: Color = Color::Rgb(255, 0, 255);
    /// Yellow accent for warnings.
    pub const ACCENT_YELLOW: Color = Color::Rgb(255, 255, 0);
    /// Red accent for errors.
    pub const ACCENT_RED: Color = Color::Rgb(255, 70, 70);

    /// Default full-screen background style.
    pub fn base() -> Style {
        Style::default().fg(Self::FG_NEON).bg(Self::BG)
    }

    /// Style for panel borders.
    pub fn border() -> Style {
        Style::default().fg(Self::FG_NEON).bg(Self::BG)
    }

    /// Style for panel and section titles.
    pub fn title() -> Style {
        Style::default()
            .fg(Self::FG_NEON)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for regular body text.
    pub fn text() -> Style {
        Style::default().fg(Self::FG_NEON)
    }

    /// Style for secondary or contextual text.
    pub fn dim() -> Style {
        Style::default().fg(Self::FG_DIM)
    }

    /// Style for disabled, placeholder, or low-priority text.
    pub fn muted() -> Style {
        Style::default().fg(Self::FG_MUTED)
    }

    /// Style for highlighted table rows.
    pub fn highlight_bg() -> Style {
        Style::default()
            .bg(Color::Rgb(0, 30, 0))
            .add_modifier(Modifier::BOLD)
    }

    /// Style for success or active states.
    pub fn ok() -> Style {
        Style::default()
            .fg(Self::FG_NEON)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for warning states.
    pub fn warn() -> Style {
        Style::default()
            .fg(Self::ACCENT_YELLOW)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for error states.
    pub fn error() -> Style {
        Style::default()
            .fg(Self::ACCENT_RED)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for informational text, equivalent to `dim`.
    pub fn info() -> Style {
        Style::default().fg(Self::FG_DIM)
    }

    /// Style for cyan-accented interactive elements.
    pub fn accent_cyan() -> Style {
        Style::default()
            .fg(Self::ACCENT_CYAN)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for magenta-accented completion indicators.
    pub fn accent_magenta() -> Style {
        Style::default()
            .fg(Self::ACCENT_MAGENTA)
            .add_modifier(Modifier::BOLD)
    }
}