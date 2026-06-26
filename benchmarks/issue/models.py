"""Model construction for the orchestra trainer, the PyTorch accuracy reference,
and the single-process PyTorch *training* baseline.

A model is a typed list of `Conv`/`MaxPool`/`Dense` layers (see `suites.py`).
Spatial dims are propagated from `input_dim`; the same ordering drives the
safetensors `layer_N` scheme so trained weights map cleanly onto the PyTorch
reference. `MaxPool` occupies a layer index but carries no weights.
"""

from __future__ import annotations

from .suites import MODELS, Activation, Conv, Dense, Loss, MaxPool, Model, TrainingConfig

# Convergence window for the baseline's early stopping. Mirrors the orchestrator's
# ConvergenceTracker (orchestrator/src/configs/adapter.rs: GreaterThanOneUsize::new(3)).
EARLY_STOP_WINSIZE = 3


def _resolve(model):
    return MODELS[model] if isinstance(model, str) else model


def _propagate(model: Model):
    """Yield (index, layer, in_dim, out_dim); dims are (C, H, W) or flat int."""
    c, h, w = model.input_dim
    flat = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv):
            out_h = (h + 2 * layer.padding - layer.kernel) // layer.stride + 1
            out_w = (w + 2 * layer.padding - layer.kernel) // layer.stride + 1
            yield i, layer, (c, h, w), (layer.filters, out_h, out_w)
            c, h, w = layer.filters, out_h, out_w
        elif isinstance(layer, MaxPool):
            out_h = (h + 2 * layer.padding - layer.size) // layer.stride + 1
            out_w = (w + 2 * layer.padding - layer.size) // layer.stride + 1
            yield i, layer, (c, h, w), (c, out_h, out_w)
            h, w = out_h, out_w
        else:  # Dense
            in_flat = flat if flat is not None else c * h * w
            yield i, layer, in_flat, layer.out
            flat = layer.out


# ── Orchestra model ──────────────────────────────────────────────────────────

def build_orchestra_model(model):
    from orchestra import Sequential
    from orchestra.activations import ReLU, Sigmoid, Softmax, Tanh
    from orchestra.arch import Conv2d
    from orchestra.arch import Dense as OrchDense
    from orchestra.arch import MaxPooling
    from orchestra.initialization import Xavier

    act_map = {Activation.SIGMOID: Sigmoid, Activation.TANH: Tanh,
               Activation.SOFTMAX: Softmax, Activation.RELU: ReLU}

    def act(a):
        return act_map[a]() if a is not None else None

    model = _resolve(model)
    layers = []
    for _, layer, in_dim, _ in _propagate(model):
        if isinstance(layer, Conv):
            c, h, w = in_dim
            layers.append(Conv2d(input_dim=(c, h, w),
                                 kernel_dim=(layer.filters, c, layer.kernel),
                                 stride=layer.stride, padding=layer.padding,
                                 init=Xavier(), act_fn=act(layer.act)))
        elif isinstance(layer, MaxPool):
            layers.append(MaxPooling(input_dim=in_dim, filter_size=layer.size,
                                     stride=layer.stride, padding=layer.padding))
        else:
            layers.append(OrchDense(layer.out, Xavier(), act(layer.act)))
    return Sequential(layers)


# ── PyTorch reference (shared by accuracy eval and baseline training) ─────────

def _torch_act(activation):
    import torch
    return {
        Activation.SIGMOID: torch.sigmoid,
        Activation.TANH: torch.tanh,
        Activation.RELU: torch.relu,
        Activation.SOFTMAX: lambda x: torch.softmax(x, dim=-1),
    }.get(activation)


def _torch_modules(model: Model):
    """Build the nn module list paired with each step's activation."""
    import torch.nn as nn

    steps = []  # (kind, module, activation)
    for _, layer, in_dim, out_dim in _propagate(model):
        if isinstance(layer, Conv):
            mod = nn.Conv2d(in_dim[0], layer.filters, layer.kernel,
                            stride=layer.stride, padding=layer.padding)
            steps.append(("conv", mod, layer.act))
        elif isinstance(layer, MaxPool):
            steps.append(("pool", nn.MaxPool2d(layer.size, stride=layer.stride), None))
        else:
            steps.append(("dense", nn.Linear(in_dim, out_dim), layer.act))
    return steps


def _build_net(model: Model, steps):
    import torch.nn as nn

    in_c, in_h, in_w = model.input_dim

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = nn.ModuleList([m for _, m, _ in steps])

        def forward(self, x, logits=False):
            """`logits=True` skips the final activation so cross-entropy can fold
            the softmax into the loss (no double normalization)."""
            x = x.view(-1, in_c, in_h, in_w)
            flattened = False
            last = len(steps) - 1
            for i, (kind, mod, activation) in enumerate(steps):
                if kind == "dense" and not flattened:
                    x = x.flatten(start_dim=1)
                    flattened = True
                x = mod(x)
                if activation is not None and not (logits and i == last):
                    x = _torch_act(activation)(x)
            return x

    return Net()


def build_torch_ref(model, state_dict):
    """Load trained orchestra weights into a PyTorch module for accuracy eval.
    Only used for argmax accuracy, so the final activation is irrelevant."""
    import torch

    model = _resolve(model)
    steps = _torch_modules(model)
    for (kind, mod, _), (i, _, _, _) in zip(steps, _propagate(model)):
        if kind == "conv":
            with torch.no_grad():
                mod.weight.copy_(state_dict[f"layer_{i}.weight"])
                mod.bias.copy_(state_dict[f"layer_{i}.bias"])
        elif kind == "dense":
            with torch.no_grad():
                mod.weight.copy_(state_dict[f"layer_{i}.weight"].T)
                mod.bias.copy_(state_dict[f"layer_{i}.bias"])
    return _build_net(model, steps).eval()


def train_torch_baseline(model, train_x, train_y, test_x, test_y, cfg: TrainingConfig):
    """Single-process PyTorch training of the same architecture and recipe.

    Reference line for the convergence plots: in principle this should match a
    standard parameter server. Returns (loss_history, accuracy).

    Cross-entropy is computed on logits via `cross_entropy` (the final softmax is
    folded into the loss) — mathematically the softmax-then-cross-entropy the
    orchestra trainer applies, without normalizing the distribution twice.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    model = _resolve(model)
    torch.manual_seed(cfg.seed)

    net = _build_net(model, _torch_modules(model)).train()
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr)

    xt = torch.tensor(np.asarray(train_x, dtype=np.float32))
    yt = torch.tensor(np.asarray(train_y, dtype=np.float32))
    targets = yt.argmax(dim=1)
    n = xt.shape[0]
    use_ce = model.loss_fn == Loss.CROSS_ENTROPY
    gen = torch.Generator().manual_seed(cfg.seed)

    # Early stopping mirrors the orchestrator's ConvergenceTracker exactly: stop
    # after EARLY_STOP_WINSIZE consecutive epochs whose loss delta stays within
    # the tolerance, so the baseline and the distributed runs use the same rule.
    tol = cfg.early_stopping_tolerance
    last, stable = None, 0

    loss_history = []
    for _ in range(cfg.max_epochs):
        perm = torch.randperm(n, generator=gen)
        epoch_loss, batches = 0.0, 0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            opt.zero_grad()
            logits = net(xt[idx], logits=True)
            loss = (F.cross_entropy(logits, targets[idx]) if use_ce
                    else F.mse_loss(logits, yt[idx]))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        avg = epoch_loss / max(batches, 1)
        loss_history.append(avg)

        if tol is not None:
            if last is not None and abs(last - avg) > tol:
                stable = 0
            stable += 1
            last = avg
            if stable == EARLY_STOP_WINSIZE:
                break

    net.eval()
    with torch.no_grad():
        xtt = torch.tensor(np.asarray(test_x, dtype=np.float32))
        ytt = torch.tensor(np.asarray(test_y, dtype=np.float32)).argmax(dim=1)
        accuracy = (net(xtt, logits=True).argmax(dim=1) == ytt).float().mean().item()
    return loss_history, accuracy
