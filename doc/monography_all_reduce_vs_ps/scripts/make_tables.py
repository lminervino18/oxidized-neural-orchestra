#!/usr/bin/env python3
"""LaTeX tables (booktabs + siunitx) built from results/processed/*.csv + the model registry.

Each table degrades gracefully if data is missing. Tables are written to tables/
and \\input by main.tex. Numeric tables come from the runs; the configs and the
decision matrix are design/qualitative and are hardcoded from the observed
evidence (kept short to fit an IEEE column).

Terminology (per reviewers):
  * Captions do NOT hardcode "Tabla" -- the paper preamble handles the naming.
  * The mlp_* models are dense networks, NOT MLPs -> shown as "Densa S/M/L".
  * Dense layers are rendered as "Dense" (never "FC").
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "results" / "processed"
TAB = ROOT / "tables"
sys.path.insert(0, str(ROOT / "scripts"))
import ono_harness as oh  # noqa: E402

PRETTY = {"all_reduce": "AR", "parameter_server": "PS", "strategy_switch": "SS",
          "baseline": "Baseline"}

# Human-facing dataset names (the CSV keys are lowercase snake_case).
DATASET_NAMES = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST",
                 "emnist": "EMNIST"}

# The mlp_* models are dense (fully-connected) networks, NOT MLPs in the paper's
# terminology -> present them as "Densa S/M/L".
MODEL_NAMES = {"nielsen": "Nielsen", "mlp_small": "Densa S",
               "mlp_medium": "Densa M", "mlp_large": "Densa L"}
# Order used for the models table (used models only; lenet5 is unused -> dropped).
MODEL_ORDER = ["nielsen", "mlp_small", "mlp_medium", "mlp_large"]
# FashionMNIST is the primary dataset in the results table, MNIST second.
DATASET_ORDER = {"fashion_mnist": 0, "mnist": 1}


def _rows(name):
    p = PROC / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _thin(n):
    """Integer with a thin-space thousands separator (28000 -> 28\\,000)."""
    try:
        return f"{int(float(n)):,}".replace(",", "\\,")
    except (TypeError, ValueError):
        return "--"


def _shape(s):
    """'28x28x1' -> '$28\\times28\\times1$'."""
    parts = [p for p in str(s).split("x") if p]
    return "$" + r"\times".join(parts) + "$" if parts else "--"


def _arch_str(model):
    """Layer sequence as a compact flow; Dense layers render as 'Dense' (not FC)."""
    m = oh.MODELS[model]
    parts = []
    for layer in m.layers:
        cn = type(layer).__name__
        if cn == "Conv":
            parts.append(f"Conv{getattr(layer, 'filters', '')}@{getattr(layer, 'kernel', '')}")
        elif cn == "MaxPool":
            parts.append("Pool")
        elif cn == "Dense":
            parts.append(f"Dense{getattr(layer, 'out', '')}")
    return r" $\to$ ".join(parts) if parts else "--"


def write_datasets():
    body = []
    for r in _rows("dataset_summary.csv"):
        name = DATASET_NAMES.get(r["dataset"], r["dataset"])
        body.append(rf"{name} & {_thin(r['train_size'])} & {_thin(r['test_size'])} & "
                    rf"{_shape(r['input_shape'])} & {r['classes']}\\")
    if not body:
        body = [r"\multicolumn{5}{c}{(sin datos)}\\"]
    lines = [r"\begin{table}[H]\centering",
             r"\caption{Conjuntos de datos utilizados en la evaluaci\'on.}\label{tab:datasets}",
             r"\footnotesize",
             r"\begin{tabular}{lrrcc}\toprule",
             r"Dataset & Train & Test & Forma & Clases\\\midrule",
             *body, r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "datasets.tex").write_text("\n".join(lines) + "\n")


def write_models():
    summ = {r["model"]: r for r in _rows("model_summary.csv")}
    body = []
    for name in MODEL_ORDER:
        row = summ.get(name)
        if row is not None and str(row.get("used")).lower() != "true":
            continue
        try:
            params = int(float(row["params"])) if row else oh.param_count(name)
        except Exception:
            params = None
        pstr = rf"\num{{{params}}}" if params is not None else "--"
        body.append(rf"{MODEL_NAMES.get(name, name)} & {_arch_str(name)} & {pstr}\\")
    if not body:
        body = [r"\multicolumn{3}{c}{(sin datos)}\\"]
    lines = [r"\begin{table}[H]\centering",
             r"\caption{Modelos empleados y su tama\~no aproximado. Las redes Densa "
             r"S/M/L son totalmente conectadas.}\label{tab:models}",
             r"\footnotesize",
             r"\begin{tabular}{llr}\toprule",
             r"Modelo & Arquitectura & Par\'ametros\\\midrule",
             *body, r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "models.tex").write_text("\n".join(lines) + "\n")


def write_configs():
    # Equal-workers design (fairness = equal workers). PS uses 2 SHARDED servers.
    # Batch efectivo = workers x 10 for the convergence configs.
    body = [r"AR & 3 & 0 & 3 & 30\\",
            r"AR & 5 & 0 & 5 & 50\\",
            r"PS & 3 & 2 & 5 & 30\\",
            r"PS & 5 & 2 & 7 & 50\\"]
    lines = [r"\begin{table}[H]\centering",
             r"\caption{Topolog\'ias distribuidas bajo el dise\~no de \emph{workers} "
             r"iguales. PS emplea 2 servidores con par\'ametros fragmentados "
             r"(\emph{sharding}); los \emph{workers} se igualan a All-Reduce para "
             r"preservar la equidad. Batch efectivo $=$ \emph{workers}$\,\times\,$10.}"
             r"\label{tab:configs}",
             r"\footnotesize",
             r"\begin{tabular}{lcccc}\toprule",
             r"Estrategia & Workers & Servidores & Nodos & Batch efectivo\\\midrule",
             *body, r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "configs.tex").write_text("\n".join(lines) + "\n")


def write_main_results():
    # Fair convergence results: labels Bg_* only. FashionMNIST (primary) first,
    # then MNIST; within a dataset ordered by workers then strategy (AR, PS).
    rows = [r for r in _rows("summary_by_strategy.csv")
            if str(r.get("label", "")).startswith("Bg_")]
    rows.sort(key=lambda r: (DATASET_ORDER.get(r["dataset"], 9),
                             _f(r["workers"]) or 0, r["strategy"]))
    body, prev_ds = [], None
    for r in rows:
        ds = r["dataset"]
        if prev_ds is not None and ds != prev_ds:
            body.append(r"\midrule")
        prev_ds = ds
        acc = _f(r.get("test_accuracy"))
        accs = f"{acc * 100:.2f}" if acc is not None else "{--}"
        t = _f(r.get("total_time_sec"))
        ts = f"{t:.1f}" if t is not None else "{--}"
        body.append(rf"{DATASET_NAMES.get(ds, ds)} & {PRETTY.get(r['strategy'], r['strategy'])} & "
                    rf"{r['workers']} & {accs} & {ts}\\")
    if not body:
        body = [r"\multicolumn{5}{c}{(sin datos)}\\"]
    lines = [r"\begin{table}[H]\centering",
             r"\caption{Resultados de convergencia justa (batch 10 por \emph{worker} "
             r"con evaluaci\'on en test) para ambas estrategias en cada conjunto de "
             r"datos.}\label{tab:main}",
             r"\footnotesize",
             r"\begin{tabular}{llcS[table-format=2.2,round-precision=2]"
             r"S[table-format=3.1,round-precision=1]}\toprule",
             r"Dataset & Estrategia & Workers & {Acc.\ test (\%)} & {Tiempo total (s)}\\\midrule",
             *body, r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "main_results.tex").write_text("\n".join(lines) + "\n")


def write_decision_matrix():
    """Qualitative decision matrix (hardcoded prose). Single-column, compact."""
    rows = [
        (r"Convergencia",
         r"Empate: misma sem\'antica s\'incrona.",
         r"Empate: misma sem\'antica s\'incrona."),
        (r"Throughput / tiempo hasta \emph{accuracy}",
         r"Favorable: sin servidor central.",
         r"Menor: sobrecarga del servidor."),
        (r"Escalado al sumar nodos",
         r"Escala bien, pendiente base.",
         r"Amortiza su costo fijo: pendiente algo mayor."),
        (r"Eficiencia de hardware",
         r"No requiere nodos servidores extra.",
         r"Requiere nodos servidores dedicados."),
        (r"Presi\'on de comunicaci\'on / modelos grandes",
         r"Sostiene la ventaja.",
         r"El \emph{sharding} acorta la brecha."),
        (r"Heterogeneidad / \emph{stragglers} / asincron\'ia",
         r"S\'incrono: sensible a \emph{stragglers}.",
         r"Ventaja estructural (asincron\'ia); no ejercitada aqu\'i."),
    ]
    body = [rf"{crit} & {ar} & {ps}\\" for crit, ar, ps in rows]
    lines = [r"\begin{table}[H]\centering",
             r"\caption{Matriz de decisi\'on cualitativa All-Reduce vs.\ Parameter "
             r"Server (acotada al entorno evaluado).}\label{tab:decision}",
             r"\footnotesize",
             r"\begin{tabular}{@{}p{0.24\columnwidth}p{0.33\columnwidth}p{0.33\columnwidth}@{}}\toprule",
             r"Criterio & All-Reduce & Parameter Server\\\midrule",
             *body, r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "decision_matrix.tex").write_text("\n".join(lines) + "\n")


def main():
    TAB.mkdir(exist_ok=True)
    for fn in (write_datasets, write_models, write_configs, write_main_results,
               write_decision_matrix):
        try:
            fn(); print("ok:", fn.__name__)
        except Exception as e:
            print("skip:", fn.__name__, "->", e)


if __name__ == "__main__":
    main()
