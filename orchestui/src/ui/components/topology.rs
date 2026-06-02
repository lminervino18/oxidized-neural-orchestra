use std::f64::consts::PI;

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        canvas::{Canvas, Circle, Context, Line as CanvasLine},
        Block, Borders,
    },
    Frame,
};

use crate::ui::{
    screens::training::{AlgorithmKind, Phase, TrainingState, WORKER_COLORS},
    theme::Theme,
};

// ── Visual constants ─────────────────────────────────────────────────────────

/// Radius of worker node circles (canvas units).
const WORKER_RADIUS: f64 = 7.0;
/// Half-width/height of server node rectangles (canvas units).
const SERVER_HALF: f64 = 6.5;
/// Color for fully-finished nodes (blue).
const COLOR_DONE: Color = Color::Rgb(40, 140, 255);
/// Color for server nodes (bright cyan).
const COLOR_SERVER: Color = Color::Rgb(0, 210, 210);
/// Color for connection lines (very dim green).
const COLOR_CONN: Color = Color::Rgb(0, 70, 0);
/// Color for animated data particles (gold).
const COLOR_PARTICLE: Color = Color::Rgb(255, 200, 0);
/// Animation: ms for a particle to complete one full connection traverse.
const ANIM_PERIOD_MS: u128 = 1800;

// ── Public entry point ───────────────────────────────────────────────────────

/// Draws the topology visualization screen.
///
/// Shows workers and servers as graphical nodes with animated data-flow arrows.
/// Layout adapts to the training algorithm: ring for AllReduce, bipartite for PS,
/// and ring-with-banner for StrategySwitch.
pub fn draw_topology(f: &mut Frame, area: Rect, state: &TrainingState) {
    let title = topology_title(state);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border())
        .title(title)
        .title_style(Theme::title());

    let elapsed_ms = state.started_at.elapsed().as_millis();
    let is_active = state.is_active();
    let phase = state.phase;

    let (nodes, conns) = compute_layout(state);

    let canvas = Canvas::default()
        .block(block)
        .x_bounds([0.0, 200.0])
        .y_bounds([0.0, 100.0])
        .paint(|ctx| {
            draw_connections(ctx, &nodes, &conns, elapsed_ms, is_active, phase);
            draw_nodes(ctx, state, &nodes, phase);
            draw_labels(ctx, state, &nodes, phase);
            draw_info_text(ctx, state, &nodes);
        });

    f.render_widget(canvas, area);
}

// ── Title ────────────────────────────────────────────────────────────────────

fn topology_title(state: &TrainingState) -> Line<'static> {
    let algo_span = match state.algorithm {
        AlgorithmKind::AllReduce => Span::styled(" ALL-REDUCE ", Theme::accent_cyan()),
        AlgorithmKind::ParameterServer => Span::styled(" PARAMETER SERVER ", Theme::accent_cyan()),
        AlgorithmKind::StrategySwitch => Span::styled(" STRATEGY SWITCH ", Theme::warn()),
    };

    let hint = Span::styled(
        "  [v] dashboard  [←/→] worker ",
        Theme::muted(),
    );

    Line::from(vec![
        Span::styled(" Topology — ", Theme::title()),
        algo_span,
        hint,
    ])
}

// ── Layout computation ────────────────────────────────────────────────────────

struct NodeInfo {
    x: f64,
    y: f64,
    label: String,
    is_server: bool,
    /// index into TrainingState.workers (None for servers)
    worker_idx: Option<usize>,
}

struct Connection {
    from_idx: usize, // index into nodes vec
    to_idx: usize,
}

fn compute_layout(state: &TrainingState) -> (Vec<NodeInfo>, Vec<Connection>) {
    match state.algorithm {
        AlgorithmKind::AllReduce | AlgorithmKind::StrategySwitch => {
            allreduce_layout(state.workers_total)
        }
        AlgorithmKind::ParameterServer => {
            ps_layout(state.workers_total, state.servers_total)
        }
    }
}

/// Circular ring layout for AllReduce.
fn allreduce_layout(n_workers: usize) -> (Vec<NodeInfo>, Vec<Connection>) {
    let n = n_workers.max(1);
    let center = (100.0_f64, 52.0_f64);
    let orbit = match n {
        1 => 0.0,
        2 => 20.0,
        3..=4 => 28.0,
        _ => 36.0,
    };

    let nodes: Vec<NodeInfo> = (0..n)
        .map(|i| {
            let angle = (i as f64) * 2.0 * PI / (n as f64) - PI / 2.0;
            NodeInfo {
                x: center.0 + orbit * angle.cos(),
                y: center.1 + orbit * angle.sin() * 0.65,
                label: format!("W{i}"),
                is_server: false,
                worker_idx: Some(i),
            }
        })
        .collect();

    let conns: Vec<Connection> = (0..n)
        .map(|i| Connection {
            from_idx: i,
            to_idx: (i + 1) % n,
        })
        .collect();

    (nodes, conns)
}

/// Bipartite layout for ParameterServer.
fn ps_layout(n_workers: usize, n_servers: usize) -> (Vec<NodeInfo>, Vec<Connection>) {
    let nw = n_workers.max(1);
    let ns = n_servers.max(1);

    let worker_nodes: Vec<NodeInfo> = (0..nw)
        .map(|i| NodeInfo {
            x: 55.0,
            y: linear_spread(i, nw, 18.0, 85.0),
            label: format!("W{i}"),
            is_server: false,
            worker_idx: Some(i),
        })
        .collect();

    let server_nodes: Vec<NodeInfo> = (0..ns)
        .map(|j| NodeInfo {
            x: 145.0,
            y: linear_spread(j, ns, 18.0, 85.0),
            label: format!("S{j}"),
            is_server: true,
            worker_idx: None,
        })
        .collect();

    let mut conns = Vec::new();
    for wi in 0..nw {
        for sj in 0..ns {
            let server_node_idx = nw + sj;
            // worker → server (gradient push)
            conns.push(Connection { from_idx: wi, to_idx: server_node_idx });
            // server → worker (param pull)
            conns.push(Connection { from_idx: server_node_idx, to_idx: wi });
        }
    }

    let mut nodes = worker_nodes;
    nodes.extend(server_nodes);
    (nodes, conns)
}

/// Evenly spreads `n` positions between `lo` and `hi`.
fn linear_spread(i: usize, n: usize, lo: f64, hi: f64) -> f64 {
    if n <= 1 {
        (lo + hi) * 0.5
    } else {
        lo + (i as f64) * (hi - lo) / (n - 1) as f64
    }
}

// ── Drawing helpers ───────────────────────────────────────────────────────────

fn draw_connections(
    ctx: &mut Context,
    nodes: &[NodeInfo],
    conns: &[Connection],
    elapsed_ms: u128,
    is_active: bool,
    phase: Phase,
) {
    // Draw connection lines first (dimmest layer)
    for conn in conns {
        let a = &nodes[conn.from_idx];
        let b = &nodes[conn.to_idx];
        ctx.draw(&CanvasLine {
            x1: a.x,
            y1: a.y,
            x2: b.x,
            y2: b.y,
            color: COLOR_CONN,
        });
    }

    // Animated particles (only while training is active)
    if !is_active || phase == Phase::Finished || phase == Phase::Error {
        return;
    }

    for (ci, conn) in conns.iter().enumerate() {
        let a = &nodes[conn.from_idx];
        let b = &nodes[conn.to_idx];

        // Each connection gets a staggered phase offset so particles don't
        // all arrive at the destination at the same time.
        let offset = (ci as u128 * ANIM_PERIOD_MS) / conns.len().max(1) as u128;
        let t = ((elapsed_ms + offset) % ANIM_PERIOD_MS) as f64 / ANIM_PERIOD_MS as f64;

        // Particle position along the line
        let px = a.x + t * (b.x - a.x);
        let py = a.y + t * (b.y - a.y);

        ctx.draw(&Circle {
            x: px,
            y: py,
            radius: 1.8,
            color: COLOR_PARTICLE,
        });
    }
}

fn draw_nodes(
    ctx: &mut Context,
    state: &TrainingState,
    nodes: &[NodeInfo],
    phase: Phase,
) {
    for node in nodes {
        let color = node_color(node, state, phase);

        if node.is_server {
            draw_server_shape(ctx, node.x, node.y, color);
        } else {
            draw_worker_shape(ctx, node.x, node.y, color);
        }
    }
}

/// Worker: three concentric circles → dense braille fill = visually "filled" circle.
fn draw_worker_shape(ctx: &mut Context, x: f64, y: f64, color: Color) {
    for radius in [WORKER_RADIUS, WORKER_RADIUS * 0.72, WORKER_RADIUS * 0.44] {
        ctx.draw(&Circle { x, y, radius, color });
    }
}

/// Server: a rectangle made from four CanvasLines (box drawing).
fn draw_server_shape(ctx: &mut Context, x: f64, y: f64, color: Color) {
    let (hw, hh) = (SERVER_HALF, SERVER_HALF * 0.7);
    // Sides
    ctx.draw(&CanvasLine { x1: x - hw, y1: y + hh, x2: x + hw, y2: y + hh, color }); // top
    ctx.draw(&CanvasLine { x1: x - hw, y1: y - hh, x2: x + hw, y2: y - hh, color }); // bottom
    ctx.draw(&CanvasLine { x1: x - hw, y1: y - hh, x2: x - hw, y2: y + hh, color }); // left
    ctx.draw(&CanvasLine { x1: x + hw, y1: y - hh, x2: x + hw, y2: y + hh, color }); // right
    // Inner fill: two inner rectangles
    let (iw, ih) = (hw * 0.6, hh * 0.6);
    ctx.draw(&CanvasLine { x1: x - iw, y1: y + ih, x2: x + iw, y2: y + ih, color });
    ctx.draw(&CanvasLine { x1: x - iw, y1: y - ih, x2: x + iw, y2: y - ih, color });
    ctx.draw(&CanvasLine { x1: x - iw, y1: y - ih, x2: x - iw, y2: y + ih, color });
    ctx.draw(&CanvasLine { x1: x + iw, y1: y - ih, x2: x + iw, y2: y + ih, color });
}

fn draw_labels(
    ctx: &mut Context,
    state: &TrainingState,
    nodes: &[NodeInfo],
    phase: Phase,
) {
    for node in nodes {
        let color = node_color(node, state, phase);
        let style = Style::default().fg(color).add_modifier(Modifier::BOLD);
        let lx = node.x - (node.label.len() as f64) * 1.1;
        ctx.print(lx, node.y - 0.8, Span::styled(node.label.clone(), style));
    }
}

fn draw_info_text(ctx: &mut Context, state: &TrainingState, nodes: &[NodeInfo]) {
    // Architecture-specific column labels for ParameterServer
    if state.algorithm == AlgorithmKind::ParameterServer && !nodes.is_empty() {
        let w_style = Style::default().fg(Theme::FG_DIM);
        let s_style = Style::default().fg(Color::Rgb(0, 180, 180));

        ctx.print(38.0, 92.0, Span::styled("workers", w_style));
        ctx.print(130.0, 92.0, Span::styled("servers", s_style));

        // Directional flow labels
        let flow_style = Style::default().fg(Theme::FG_MUTED);
        ctx.print(82.0, 58.0, Span::styled("→ grads", flow_style));
        ctx.print(82.0, 47.0, Span::styled("← params", flow_style));
    }

    // Strategy switch banner
    if state.algorithm == AlgorithmKind::StrategySwitch {
        let banner_style = Style::default()
            .fg(Theme::ACCENT_YELLOW)
            .add_modifier(Modifier::BOLD);
        ctx.print(
            28.0,
            92.0,
            Span::styled("⟳  will switch to parameter server when converging", banner_style),
        );
    }

    // Phase overlay: when finished, print "DONE" banner
    if state.phase == Phase::Finished {
        let done_style = Style::default()
            .fg(COLOR_DONE)
            .add_modifier(Modifier::BOLD);
        ctx.print(76.0, 8.0, Span::styled("✓  TRAINING COMPLETE", done_style));
    }

    // Legend (bottom-right corner)
    draw_legend(ctx, state);
}

fn draw_legend(ctx: &mut Context, state: &TrainingState) {
    let muted = Style::default().fg(Theme::FG_MUTED);
    let done_style = Style::default().fg(COLOR_DONE);

    let base_y = 6.0;
    let base_x = 153.0;

    ctx.print(
        base_x,
        base_y + 4.5,
        Span::styled("● active worker", muted),
    );

    if state.servers_total > 0 {
        ctx.print(
            base_x,
            base_y + 2.5,
            Span::styled("■ parameter server", Style::default().fg(Color::Rgb(0, 180, 180))),
        );
    }

    ctx.print(
        base_x,
        base_y + 0.5,
        Span::styled("● done / finished", done_style),
    );
}

// ── Color resolution ──────────────────────────────────────────────────────────

fn node_color(node: &NodeInfo, state: &TrainingState, phase: Phase) -> Color {
    if node.is_server {
        return server_color(phase);
    }

    let Some(wi) = node.worker_idx else {
        return Theme::FG_DIM;
    };

    let worker = state.workers.get(wi);

    if phase == Phase::Finished {
        return COLOR_DONE;
    }

    if let Some(w) = worker {
        if w.done {
            return COLOR_DONE;
        }
    }

    if phase == Phase::Error {
        return Theme::ACCENT_RED;
    }

    WORKER_COLORS[wi % WORKER_COLORS.len()]
}

fn server_color(phase: Phase) -> Color {
    if phase == Phase::Finished {
        COLOR_DONE
    } else if phase == Phase::Error {
        Theme::ACCENT_RED
    } else {
        COLOR_SERVER
    }
}
