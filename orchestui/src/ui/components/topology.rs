use std::f64::consts::PI;

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        canvas::{Canvas, Circle, Context, Line as CanvasLine, Points},
        Block, Borders,
    },
    Frame,
};

use crate::ui::{
    screens::training::{AlgorithmKind, Phase, TrainingState, WORKER_COLORS},
    theme::Theme,
};

// ── Fixed visual constants ────────────────────────────────────────────────────

/// X coordinate of canvas center.
const CX: f64 = 100.0;
/// Y coordinate of canvas center for the node drawing zone.
const CY: f64 = 49.5;

/// Bottom boundary of the node drawing zone (above legend area).
const ZONE_LO: f64 = 20.0;
/// Top boundary of the node drawing zone (below banner / title area).
const ZONE_HI: f64 = 80.0;

/// Aspect ratio correction so circles look roughly round on typical terminals.
/// y_orbit = x_orbit × ASPECT. Higher makes the ring taller / less flat.
const ASPECT: f64 = 0.70;

/// Minimum gap between the edges of two adjacent nodes (canvas units).
const NODE_GAP: f64 = 3.0;

// ── Colors ────────────────────────────────────────────────────────────────────

const COLOR_DONE: Color = Color::Rgb(40, 140, 255);
const COLOR_SERVER: Color = Color::Rgb(0, 210, 210);
const COLOR_CONN: Color = Color::Rgb(0, 55, 0);
const COLOR_PARTICLE: Color = Color::Rgb(255, 200, 0);
const ANIM_PERIOD_MS: u128 = 1800;

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn draw_topology(f: &mut Frame, area: ratatui::layout::Rect, state: &TrainingState) {
    let (nodes, conns, radius) = compute_layout(state);
    let elapsed_ms = state.started_at.elapsed().as_millis();
    let is_active = state.is_active();
    let phase = state.phase;

    // Canvas units per text row: lets us space the node's inner lines exactly
    // one row apart regardless of terminal height.
    let inner_h = area.height.saturating_sub(2).max(1) as f64;
    let row = 100.0 / inner_h;

    let canvas = Canvas::default()
        .block(topology_block(state))
        .background_color(Theme::BG)
        .x_bounds([0.0, 200.0])
        .y_bounds([0.0, 100.0])
        .paint(|ctx| {
            draw_connections(ctx, &nodes, &conns, elapsed_ms, is_active, phase);
            draw_shapes(ctx, state, &nodes, phase, radius);
            draw_node_content(ctx, state, &nodes, phase, radius, row);
            draw_static_labels(ctx, state, &nodes, radius);
        });

    f.render_widget(canvas, area);
}

fn topology_block(state: &TrainingState) -> Block<'static> {
    let (algo, algo_style) = match state.algorithm {
        AlgorithmKind::AllReduce => (" ALL-REDUCE ", Theme::accent_cyan()),
        AlgorithmKind::ParameterServer => (" PARAMETER SERVER ", Theme::accent_cyan()),
        AlgorithmKind::StrategySwitch if state.ps_server_count > 0 => {
            (" STRATEGY SWITCH → PS ", Theme::accent_magenta())
        }
        AlgorithmKind::StrategySwitch => (" STRATEGY SWITCH ", Theme::warn()),
    };

    let title = Line::from(vec![
        Span::styled(" Topology — ", Theme::title()),
        Span::styled(algo, algo_style),
        Span::styled("  [v] dashboard  [←/→] worker ", Theme::muted()),
    ]);

    Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border())
        .title(title)
        .title_style(Theme::title())
}

// ── Node data types ───────────────────────────────────────────────────────────

struct NodeInfo {
    x: f64,
    y: f64,
    label: String,
    is_server: bool,
    worker_idx: Option<usize>,
}

struct Connection {
    from_idx: usize,
    to_idx: usize,
}

// ── Dynamic radius helpers ────────────────────────────────────────────────────

/// Target center-to-center distance as a multiple of node diameter.
const SEPARATION: f64 = 2.1;

/// Returns `(radius, orbit)` for `n` nodes in an AllReduce ring.
///
/// Orbit is chosen so adjacent nodes are `SEPARATION` diameters apart
/// (generous visual breathing room). Radius shrinks if needed to fit within
/// the safe zone.
fn allreduce_params(n: usize) -> (f64, f64) {
    if n <= 1 {
        return (14.0, 0.0);
    }

    // Try radii from large to small; pick the largest whose ring fits the zone.
    // Small topologies get big nodes; they shrink as the ring grows, never overlapping.
    for &r in &[14.0_f64, 12.0, 10.0, 8.0, 6.5, 5.0, 4.0, 3.0, 2.5] {
        // orbit so that adjacent node centres are SEPARATION × diameter apart
        let orbit = SEPARATION * r / (PI / n as f64).sin();
        // the whole node (centre + radius) must stay inside the vertical zone
        let max_orbit = (ZONE_HI - CY - r) / ASPECT;
        if orbit <= max_orbit {
            return (r, orbit.max(12.0));
        }
    }

    let r = 2.5;
    (r, (ZONE_HI - CY - r) / ASPECT)
}

/// Returns the largest node radius that allows stacking `max_dim` nodes
/// vertically within `ZONE_LO..ZONE_HI` without overlap.
fn ps_node_radius(max_dim: usize) -> f64 {
    if max_dim <= 1 {
        return 11.0;
    }
    // `vspread` insets the centers by one radius on each side, so the real
    // center-to-center spacing is `(available - 2r) / (n - 1)`. Solving
    // `spacing >= 2r + NODE_GAP` for `r` yields this closed form, which keeps
    // nodes as large as possible while guaranteeing they never overlap.
    let available = ZONE_HI - ZONE_LO;
    let n = max_dim as f64;
    let max_r = (available - (n - 1.0) * NODE_GAP) / (2.0 * n);
    max_r.min(11.0).max(2.0)
}

// ── Layout computation ────────────────────────────────────────────────────────

fn compute_layout(state: &TrainingState) -> (Vec<NodeInfo>, Vec<Connection>, f64) {
    match state.algorithm {
        AlgorithmKind::AllReduce => allreduce_layout(state.workers_total),
        AlgorithmKind::StrategySwitch if state.ps_server_count > 0 => {
            ss_switched_layout(state)
        }
        AlgorithmKind::StrategySwitch => allreduce_layout(state.workers_total),
        AlgorithmKind::ParameterServer => ps_layout(state.workers_total, state.servers_total),
    }
}

fn allreduce_layout(n: usize) -> (Vec<NodeInfo>, Vec<Connection>, f64) {
    let n = n.max(1);
    let (radius, orbit) = allreduce_params(n);

    let nodes: Vec<NodeInfo> = (0..n)
        .map(|i| {
            let angle = (i as f64) * 2.0 * PI / (n as f64) - PI / 2.0;
            NodeInfo {
                x: CX + orbit * angle.cos(),
                y: CY + orbit * angle.sin() * ASPECT,
                label: format!("W{i}"),
                is_server: false,
                worker_idx: Some(i),
            }
        })
        .collect();

    let conns: Vec<Connection> = (0..n)
        .map(|i| Connection { from_idx: i, to_idx: (i + 1) % n })
        .collect();

    (nodes, conns, radius)
}

fn ps_layout(nw: usize, ns: usize) -> (Vec<NodeInfo>, Vec<Connection>, f64) {
    let nw = nw.max(1);
    let ns = ns.max(1);
    let radius = ps_node_radius(nw.max(ns));

    let mut nodes: Vec<NodeInfo> = (0..nw)
        .map(|i| NodeInfo {
            x: 50.0,
            y: vspread(i, nw, radius),
            label: format!("W{i}"),
            is_server: false,
            worker_idx: Some(i),
        })
        .collect();

    for j in 0..ns {
        nodes.push(NodeInfo {
            x: 150.0,
            y: vspread(j, ns, radius),
            label: format!("S{j}"),
            is_server: true,
            worker_idx: None,
        });
    }

    let mut conns = Vec::new();
    for wi in 0..nw {
        for sj in 0..ns {
            conns.push(Connection { from_idx: wi, to_idx: nw + sj });
            conns.push(Connection { from_idx: nw + sj, to_idx: wi });
        }
    }

    (nodes, conns, radius)
}

fn ss_switched_layout(state: &TrainingState) -> (Vec<NodeInfo>, Vec<Connection>, f64) {
    let active: Vec<usize> = state
        .workers
        .iter()
        .filter(|w| !w.became_server)
        .map(|w| w.id)
        .collect();

    let nw = active.len().max(1);
    let ns = state.ps_server_count.max(1);
    let radius = ps_node_radius(nw.max(ns));

    let mut nodes: Vec<NodeInfo> = active
        .iter()
        .enumerate()
        .map(|(pos, &wi)| NodeInfo {
            x: 50.0,
            y: vspread(pos, nw, radius),
            label: format!("W{wi}"),
            is_server: false,
            worker_idx: Some(wi),
        })
        .collect();

    for j in 0..ns {
        nodes.push(NodeInfo {
            x: 150.0,
            y: vspread(j, ns, radius),
            label: format!("S{j}"),
            is_server: true,
            worker_idx: None,
        });
    }

    let mut conns = Vec::new();
    for wi in 0..nw {
        for sj in 0..ns {
            conns.push(Connection { from_idx: wi, to_idx: nw + sj });
            conns.push(Connection { from_idx: nw + sj, to_idx: wi });
        }
    }

    (nodes, conns, radius)
}

/// Spreads `n` node centers evenly in `ZONE_LO+r .. ZONE_HI-r`.
fn vspread(i: usize, n: usize, r: f64) -> f64 {
    let lo = ZONE_LO + r;
    let hi = ZONE_HI - r;
    if n <= 1 {
        (lo + hi) * 0.5
    } else {
        lo + (i as f64) * (hi - lo) / (n - 1) as f64
    }
}

// ── Connections + particles ───────────────────────────────────────────────────

fn draw_connections(
    ctx: &mut Context,
    nodes: &[NodeInfo],
    conns: &[Connection],
    elapsed_ms: u128,
    is_active: bool,
    phase: Phase,
) {
    for conn in conns {
        let a = &nodes[conn.from_idx];
        let b = &nodes[conn.to_idx];
        ctx.draw(&CanvasLine { x1: a.x, y1: a.y, x2: b.x, y2: b.y, color: COLOR_CONN });
    }

    if !is_active || phase == Phase::Finished || phase == Phase::Error {
        return;
    }

    let n = conns.len().max(1);
    for (ci, conn) in conns.iter().enumerate() {
        let a = &nodes[conn.from_idx];
        let b = &nodes[conn.to_idx];
        let offset = (ci as u128 * ANIM_PERIOD_MS) / n as u128;
        let t = ((elapsed_ms + offset) % ANIM_PERIOD_MS) as f64 / ANIM_PERIOD_MS as f64;
        let pr = (1.5_f64).min(0.18 * radius_estimate(a, b));
        ctx.draw(&Circle {
            x: a.x + t * (b.x - a.x),
            y: a.y + t * (b.y - a.y),
            radius: pr,
            color: COLOR_PARTICLE,
        });
    }
}

fn radius_estimate(a: &NodeInfo, b: &NodeInfo) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (dx * dx + dy * dy).sqrt() * 0.05
}

// ── Node shape drawing ────────────────────────────────────────────────────────

fn draw_shapes(
    ctx: &mut Context,
    state: &TrainingState,
    nodes: &[NodeInfo],
    phase: Phase,
    radius: f64,
) {
    for node in nodes {
        let color = node_color(node, state, phase);
        if node.is_server {
            let hw = radius;
            let hh = radius * 0.68;
            let x = node.x;
            let y = node.y;
            ctx.draw(&CanvasLine { x1: x - hw, y1: y + hh, x2: x + hw, y2: y + hh, color });
            ctx.draw(&CanvasLine { x1: x - hw, y1: y - hh, x2: x + hw, y2: y - hh, color });
            ctx.draw(&CanvasLine { x1: x - hw, y1: y - hh, x2: x - hw, y2: y + hh, color });
            ctx.draw(&CanvasLine { x1: x + hw, y1: y - hh, x2: x + hw, y2: y + hh, color });
        } else {
            draw_dense_circle(ctx, node.x, node.y, radius, color);
        }
    }
}

/// Draws a circle outline as a dense ring of points so the contour reads as a
/// compact, continuous line with no gaps at any radius.
fn draw_dense_circle(ctx: &mut Context, cx: f64, cy: f64, r: f64, color: Color) {
    let steps = ((2.0 * PI * r) * 3.0).ceil().max(60.0) as usize;
    let coords: Vec<(f64, f64)> = (0..steps)
        .map(|i| {
            let a = (i as f64) / (steps as f64) * 2.0 * PI;
            (cx + r * a.cos(), cy + r * a.sin())
        })
        .collect();
    ctx.draw(&Points { coords: &coords, color });
}

// ── Node text content (scaled to radius) ─────────────────────────────────────

fn draw_node_content(
    ctx: &mut Context,
    state: &TrainingState,
    nodes: &[NodeInfo],
    phase: Phase,
    radius: f64,
    row: f64,
) {
    for node in nodes {
        let color = node_color(node, state, phase);
        let bold = Style::default().fg(color).add_modifier(Modifier::BOLD);

        // Text tiers depend both on the node radius and on how many text rows
        // actually fit inside it, so lines never overflow the shape on short
        // terminals (where a single row spans many canvas units).
        let half_rows = radius / row;
        let show_epoch = radius >= 7.0 && half_rows >= 1.5;
        let show_loss = radius >= 5.0 && half_rows >= 0.9;
        let show_label = radius >= 3.5;

        if !show_label {
            continue;
        }

        // The lines sit exactly one text row apart, centered on the node:
        // loss stays at the center, the label one row above, epoch one below.
        let label_dy = if show_loss { row } else { 0.0 };
        let loss_dy = 0.0;
        let epoch_dy = -row;

        let label = node.label.clone();
        ctx.print(lcx(node.x, label.len()), node.y + label_dy, Span::styled(label, bold));

        if let Some(wi) = node.worker_idx {
            if let Some(w) = state.workers.get(wi) {
                let dim = Style::default().fg(darken(color));

                if show_loss {
                    let loss = fmt_short(w.last_loss);
                    ctx.print(lcx(node.x, loss.len()), node.y + loss_dy, Span::styled(loss, dim));
                }

                if show_epoch {
                    let ep = format!("{}/{}", w.epochs_done, state.max_epochs);
                    ctx.print(lcx(node.x, ep.len()), node.y + epoch_dy, Span::styled(ep, dim));
                }
            }
        }
    }
}

/// Approximate canvas x that horizontally centers text of `len` chars at `x`.
fn lcx(x: f64, len: usize) -> f64 {
    x - (len as f64) * 0.85
}

fn fmt_short(v: Option<f64>) -> String {
    match v {
        None => "—".into(),
        Some(l) if l.abs() < 1e-4 => format!("{l:.2e}"),
        Some(l) => format!("{l:.4}"),
    }
}

fn darken(c: Color) -> Color {
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(
            (r as u16 * 7 / 10) as u8,
            (g as u16 * 7 / 10) as u8,
            (b as u16 * 7 / 10) as u8,
        ),
        other => other,
    }
}

// ── Static overlay text ───────────────────────────────────────────────────────

fn draw_static_labels(ctx: &mut Context, state: &TrainingState, _nodes: &[NodeInfo], _radius: f64) {
    let algo = state.algorithm;
    let ps_active = algo == AlgorithmKind::ParameterServer
        || (algo == AlgorithmKind::StrategySwitch && state.ps_server_count > 0);

    // Column type labels (well above ZONE_HI)
    if ps_active {
        ctx.print(32.0, 86.0, Span::styled("workers", Style::default().fg(Theme::FG_DIM)));
        ctx.print(132.0, 86.0, Span::styled("servers", Style::default().fg(COLOR_SERVER)));

        // Flow direction arrows: placed at mid-height, in the gap between columns
        let mid_x = 82.0;
        let mid_y = CY;
        ctx.print(mid_x, mid_y + 2.0, Span::styled("→ grads",  Style::default().fg(Theme::FG_MUTED)));
        ctx.print(mid_x, mid_y - 2.0, Span::styled("← params", Style::default().fg(Theme::FG_MUTED)));
    }

    // StrategySwitch banner (top, above column labels)
    if algo == AlgorithmKind::StrategySwitch && state.ps_server_count == 0 {
        ctx.print(
            10.0, 92.0,
            Span::styled(
                "⟳  all-reduce → will switch to PS when gradients converge",
                Style::default().fg(Theme::ACCENT_YELLOW).add_modifier(Modifier::BOLD),
            ),
        );
    }

    // Finished banner (bottom zone, above legend)
    if state.phase == Phase::Finished {
        ctx.print(
            64.0, 14.0,
            Span::styled(
                "✓  TRAINING COMPLETE",
                Style::default().fg(COLOR_DONE).add_modifier(Modifier::BOLD),
            ),
        );
    }

    // Legend (bottom-right, below node zone)
    draw_legend(ctx, state);
}

fn draw_legend(ctx: &mut Context, state: &TrainingState) {
    let base_x = 145.0;
    let base_y = 11.0;

    ctx.print(base_x, base_y + 5.0, Span::styled("● active worker", Style::default().fg(Theme::FG_MUTED)));

    let ps_active = state.algorithm == AlgorithmKind::ParameterServer
        || (state.algorithm == AlgorithmKind::StrategySwitch && state.ps_server_count > 0);

    if ps_active || state.servers_total > 0 {
        ctx.print(base_x, base_y + 2.5, Span::styled("■ parameter server", Style::default().fg(COLOR_SERVER)));
    }

    ctx.print(base_x, base_y, Span::styled("● done (blue)", Style::default().fg(COLOR_DONE)));
}

// ── Color resolution ──────────────────────────────────────────────────────────

fn node_color(node: &NodeInfo, state: &TrainingState, phase: Phase) -> Color {
    if node.is_server {
        return match phase {
            Phase::Finished => COLOR_DONE,
            Phase::Error => Theme::ACCENT_RED,
            _ => COLOR_SERVER,
        };
    }

    let wi = match node.worker_idx {
        Some(i) => i,
        None => return Theme::FG_DIM,
    };

    if phase == Phase::Finished {
        return COLOR_DONE;
    }
    if phase == Phase::Error {
        return Theme::ACCENT_RED;
    }

    let worker = state.workers.get(wi);

    if worker.map_or(false, |w| w.done) {
        return COLOR_DONE;
    }

    if worker.map_or(true, |w| w.last_loss.is_none()) && phase == Phase::Connecting {
        return Theme::FG_MUTED;
    }

    WORKER_COLORS[wi % WORKER_COLORS.len()]
}
