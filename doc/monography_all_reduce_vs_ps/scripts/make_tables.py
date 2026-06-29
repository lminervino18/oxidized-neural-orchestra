#!/usr/bin/env python3
"""LaTeX tables (booktabs) built from results/processed/*.csv + the model registry.

Each table degrades gracefully if data is missing. Tables are written to tables/
and \\input by main.tex. Numeric tables come from the runs; the decision matrix is
qualitative and is filled from the observed evidence (kept short to fit a column).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "results" / "processed"
TAB = ROOT / "tables"
sys.path.insert(0, str(ROOT / "scripts"))
import ono_harness as oh  # noqa: E402

PRETTY = {"all_reduce": "AR", "parameter_server": "PS", "strategy_switch": "SS",
          "baseline": "Baseline"}


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


def _arch_str(model):
    m = oh.resolve_model(model) if hasattr(oh, "resolve_model") else oh.MODELS[model]
    parts = []
    for layer in m.layers:
        cn = type(layer).__name__
        if cn == "Conv":
            parts.append(f"Conv{getattr(layer,'filters','')}@{getattr(layer,'kernel','')}")
        elif cn == "MaxPool":
            parts.append("Pool")
        elif cn == "Dense":
            parts.append(f"FC{getattr(layer,'out','')}")
    return "--".join(parts) if parts else "--"


def write_datasets():
    has_emnist = (ROOT / "datasets" / "emnist").exists()
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Datasets utilizados.}\label{tab:datasets}",
             r"\footnotesize",
             r"\begin{tabular}{lrrcc}\toprule",
             r"Dataset & Train & Test & Forma & Clases\\\midrule",
             r"MNIST & 60\,000 & 10\,000 & $28\times28$ & 10\\"]
    if has_emnist:
        lines.append(r"EMNIST & --- & --- & $28\times28$ & --\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "datasets.tex").write_text("\n".join(lines) + "\n")


def write_models():
    used = {r["model"] for r in _rows("summary_by_strategy.csv")} or {"nielsen"}
    order = ["nielsen", "lenet5", "mlp_small", "mlp_medium", "mlp_large"]
    rows = []
    for name in order:
        if name not in used or name not in oh.MODELS:
            continue
        try:
            p = oh.param_count(name)
        except Exception:
            p = None
        pstr = f"{p:,}".replace(",", "\\,") if p else "--"
        rows.append(rf"\texttt{{{name.replace('_','\\_')}}} & {_arch_str(name)} & {pstr}\\")
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Modelos y tama\~no aproximado.}\label{tab:models}",
             r"\footnotesize",
             r"\begin{tabular}{lll}\toprule",
             r"Modelo & Arquitectura & Par\'ametros\\\midrule",
             *rows, r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "models.tex").write_text("\n".join(lines) + "\n")


def write_configs():
    rows = _rows("summary_by_strategy.csv")
    seen, body = set(), []
    for r in sorted(rows, key=lambda r: (r["strategy"], _f(r["nodes"]) or 0)):
        if r.get("strategy") == "baseline":
            continue
        key = (r["strategy"], r["nodes"], r["workers"], r["servers"])
        if key in seen:
            continue
        seen.add(key)
        sync = r.get("synchronizer") or "--"
        store = r.get("store") or "--"
        body.append(rf"{PRETTY.get(r['strategy'], r['strategy'])} & {r['nodes']} & "
                    rf"{r['workers']} & {r['servers']} & {sync}/{store} & "
                    rf"{r.get('effective_global_batch') or '--'}\\")
    if not body:
        body = [r"\multicolumn{6}{c}{(sin datos)}\\"]
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Configuraciones distribuidas evaluadas.}\label{tab:configs}",
             r"\footnotesize",
             r"\begin{tabular}{lcccll}\toprule",
             r"Estr. & Nodos & W & S & sync/store & Batch ef.\\\midrule",
             *body, r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "configs.tex").write_text("\n".join(lines) + "\n")


def write_main_results():
    rows = _rows("summary_by_strategy.csv")
    body = []
    # Fair comparison only: the fixed-global-batch Exp B runs (labels Bg_*). The
    # throughput-scaling speedup lives in its own figure, not in this table.
    for r in sorted(rows, key=lambda r: (r["model"], _f(r["nodes"]) or 0, r["strategy"])):
        if r.get("model") != "nielsen" or not r.get("label", "").startswith("Bg_"):
            continue
        acc = _f(r.get("test_accuracy"))
        accs = f"{acc*100:.2f}" if acc is not None else "--"
        t = _f(r.get("total_time_sec"))
        ts = f"{t:.0f}" if t else "--"
        sps = _f(r.get("samples_per_sec"))
        spss = f"{sps:,.0f}".replace(",", "\\,") if sps else "--"
        body.append(rf"{PRETTY.get(r['strategy'], r['strategy'])} & {r['nodes']} & "
                    rf"{accs} & {ts} & {spss}\\")
    if not body:
        body = [r"\multicolumn{5}{c}{(sin datos)}\\"]
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Resultados principales en MNIST/\texttt{nielsen} (batch global fijo).}\label{tab:main}",
             r"\footnotesize",
             r"\begin{tabular}{lccrr}\toprule",
             r"Estr. & Nodos & Acc.\,(\%) & Tiempo\,(s) & samp./s\\\midrule",
             *body, r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "main_results.tex").write_text("\n".join(lines) + "\n")


def write_decision_matrix():
    """Qualitative; full page width. Edit the evidence column after analysis."""
    lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{Matriz de decisi\'on All-Reduce vs.\ Parameter Server (acotada al entorno evaluado).}\label{tab:decision}",
        r"\footnotesize",
        r"\begin{tabular}{p{0.30\textwidth}p{0.13\textwidth}p{0.49\textwidth}}\toprule",
        r"Condici\'on & Recomendada & Justificaci\'on / evidencia\\\midrule",
        r"Cl\'uster homog\'eneo, buena red, sincron\'ia, gradientes densos, evitar servidor central & All-Reduce & "
        r"Sin punto central de contenci\'on; comunicaci\'on por worker $\approx$ independiente de $W$. \emph{Evidencia:} iguala a PS en accuracy con mayor throughput y menor tiempo hasta accuracy.\\",
        r"Modelos grandes, mayor escala, sharding de par\'ametros, desacoplar workers/servers & Parameter Server & "
        r"El sharding reparte la carga de comunicaci\'on. \emph{Evidencia:} escala con mayor pendiente al sumar nodos y complet\'o el entrenamiento del MLP m\'as grande, donde All-Reduce no lo logr\'o.\\",
        r"\bottomrule\end{tabular}", r"\end{table*}"]
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
