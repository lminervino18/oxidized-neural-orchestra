use std::io;
use std::num::NonZeroUsize;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use orchestrator::{configs::*, train};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::state::session::SessionState;
use crate::ui::draw::draw;

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

/// Runs the interactive TUI backed by a real training session.
///
/// # Errors
/// Returns an error if terminal setup, rendering, or session creation fails.
pub fn run() -> Result<()> {
    let _guard = TerminalGuard::enter()?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let (model, training, workers_total) = build_training_config();

    let mut session = train(model, training)?;
    let events = session.take_events().expect("fresh session has no events");

    let mut state = SessionState::new(workers_total, events);
    let mut show_diagram = true;
    let mut show_logs = true;

    loop {
        state.tick();
        let view = state.view();

        terminal.draw(|f| draw(f, &view, show_diagram, show_logs))?;

        if event::poll(Duration::from_millis(120))? {
            match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => match k.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('d') => show_diagram = !show_diagram,
                    KeyCode::Char('l') => show_logs = !show_logs,
                    _ => {}
                },
                _ => {}
            }
        }
    }

    terminal.show_cursor()?;
    Ok(())
}

/// Returns the model config, training config, and worker count.
fn build_training_config() -> (ModelConfig, TrainingConfig<&'static str>, usize) {
    let worker_addrs = vec!["worker-0:50000", "worker-1:50000"];
    let workers_total = worker_addrs.len();

    let model = ModelConfig::Sequential {
        layers: vec![LayerConfig::Dense {
            dim: (1, 1),
            init: ParamGenConfig::Const { value: 0.0 },
            act_fn: None,
        }],
    };

    let training = TrainingConfig {
        worker_addrs,
        algorithm: AlgorithmConfig::ParameterServer {
            server_addrs: vec!["server-0:40000"],
            synchronizer: SynchronizerConfig::Barrier { barrier_size: 2 },
            store: StoreConfig::Blocking {
                shard_size: NonZeroUsize::new(1).unwrap(),
            },
        },
        dataset: DatasetConfig::Inline {
            data: vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
            x_size: 1,
            y_size: 1,
        },
        optimizer: OptimizerConfig::GradientDescent { lr: 0.1 },
        loss_fn: LossFnConfig::Mse,
        batch_size: NonZeroUsize::new(4).unwrap(),
        max_epochs: NonZeroUsize::new(100).unwrap(),
        offline_epochs: 0,
        seed: None,
    };

    (model, training, workers_total)
}