"""Model construction for both the orchestra trainer and the PyTorch reference.

A model is a list of declarative layers (see suites.MODELS). Spatial dims are
propagated from `input_dim`; the same ordering is used for the safetensors
`layer_N` scheme so the trained weights map cleanly onto the PyTorch reference.
MaxPooling occupies a layer index but carries no weights.
"""

from .suites import MODELS


def _propagate(model_name):
    """Yield (index, layer, in_dim, out_dim) where dims are (C, H, W) or flat int."""
    spec = MODELS[model_name]
    c, h, w = spec["input_dim"]
    flat = None
    for i, layer in enumerate(spec["layers"]):
        kind = layer["type"]
        if kind == "conv":
            k, s, p = layer["kernel"], layer["stride"], layer["padding"]
            out_c = layer["filters"]
            out_h = (h + 2 * p - k) // s + 1
            out_w = (w + 2 * p - k) // s + 1
            yield i, layer, (c, h, w), (out_c, out_h, out_w)
            c, h, w = out_c, out_h, out_w
        elif kind == "maxpool":
            k, s, p = layer["size"], layer["stride"], layer["padding"]
            out_h = (h + 2 * p - k) // s + 1
            out_w = (w + 2 * p - k) // s + 1
            yield i, layer, (c, h, w), (c, out_h, out_w)
            h, w = out_h, out_w
        else:  # dense
            in_flat = flat if flat is not None else c * h * w
            yield i, layer, in_flat, layer["out"]
            flat = layer["out"]


def build_orchestra_model(model_name):
    from orchestra import Sequential
    from orchestra.arch import Conv2d, Dense, MaxPooling
    from orchestra.activations import ReLU, Sigmoid, Softmax, Tanh
    from orchestra.initialization import Xavier

    def act(name):
        return {"sigmoid": Sigmoid, "tanh": Tanh, "softmax": Softmax, "relu": ReLU}[name]() if name else None

    layers = []
    for _, layer, in_dim, _ in _propagate(model_name):
        kind = layer["type"]
        if kind == "conv":
            c, h, w = in_dim
            layers.append(Conv2d(input_dim=(c, h, w),
                                 kernel_dim=(layer["filters"], c, layer["kernel"]),
                                 stride=layer["stride"], padding=layer["padding"],
                                 init=Xavier(), act_fn=act(layer.get("act"))))
        elif kind == "maxpool":
            layers.append(MaxPooling(input_dim=in_dim, filter_size=layer["size"],
                                     stride=layer["stride"], padding=layer["padding"]))
        else:
            layers.append(Dense(layer["out"], Xavier(), act(layer.get("act"))))
    return Sequential(layers)


def build_torch_ref(model_name, state_dict):
    import torch
    import torch.nn as nn

    def act(name):
        if name == "sigmoid":
            return torch.sigmoid
        if name == "tanh":
            return torch.tanh
        if name == "softmax":
            return lambda x: torch.softmax(x, dim=-1)
        return None

    steps = []
    for i, layer, in_dim, out_dim in _propagate(model_name):
        kind = layer["type"]
        if kind == "conv":
            c = in_dim[0]
            mod = nn.Conv2d(c, layer["filters"], layer["kernel"],
                            stride=layer["stride"], padding=layer["padding"])
            with torch.no_grad():
                mod.weight.copy_(state_dict[f"layer_{i}.weight"])
                mod.bias.copy_(state_dict[f"layer_{i}.bias"])
            steps.append(("conv", mod, act(layer.get("act"))))
        elif kind == "maxpool":
            steps.append(("pool", nn.MaxPool2d(layer["size"], stride=layer["stride"]), None))
        else:
            mod = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                mod.weight.copy_(state_dict[f"layer_{i}.weight"].T)
                mod.bias.copy_(state_dict[f"layer_{i}.bias"])
            steps.append(("dense", mod, act(layer.get("act"))))

    in_c, in_h, in_w = MODELS[model_name]["input_dim"]

    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = nn.ModuleList([m for _, m, _ in steps])

        def forward(self, x):
            x = x.view(-1, in_c, in_h, in_w)
            flattened = False
            for kind, mod, fn in steps:
                if kind == "dense" and not flattened:
                    x = x.flatten(start_dim=1)
                    flattened = True
                x = mod(x)
                if fn is not None:
                    x = fn(x)
            return x

    return Ref().eval()
