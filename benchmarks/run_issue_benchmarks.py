#!/usr/bin/env python3
"""Issue benchmark runner: parameter-server vs all-reduce vs strategy-switch.

Usage:
  .venv/bin/python benchmarks/run_issue_benchmarks.py
  .venv/bin/python benchmarks/run_issue_benchmarks.py --suite convergence
  .venv/bin/python benchmarks/run_issue_benchmarks.py --suite scalability --model lenet5
  .venv/bin/python benchmarks/run_issue_benchmarks.py --plots-only
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _ensure_orchestra():
    try:
        import orchestra  # noqa: F401
        return
    except ImportError:
        pass
    for venv in [REPO_ROOT / ".venv", REPO_ROOT / "orchestra-py" / ".venv"]:
        for site in (venv / "lib").glob("python*/site-packages"):
            sys.path.insert(0, str(site))
    try:
        import orchestra  # noqa: F401
    except ImportError:
        sys.exit("ERROR: orchestra not importable. Build it with: "
                 "cd orchestra-py && maturin develop --release")


def parse_args():
    from issue.suites import ALL_MODELS, ALL_SUITES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", action="append", choices=ALL_SUITES,
                   help="Suite(s) to run (default: all)")
    p.add_argument("--model", action="append", choices=ALL_MODELS,
                   help="Model(s) to run (default: all)")
    p.add_argument("--strategy", action="append",
                   choices=["parameter_server", "all_reduce", "strategy_switch"],
                   help="Strategy(ies) to run (default: all)")
    p.add_argument("--rebuild", action="store_true", help="Force a Docker image rebuild")
    p.add_argument("--keep-containers", action="store_true", help="Leave containers up at the end")
    p.add_argument("--plots-only", action="store_true",
                   help="Rebuild plots and README from history without training")
    return p.parse_args()


def main():
    args = parse_args()
    _ensure_orchestra()

    from collections import defaultdict

    from issue import results as R
    from issue.plots import plot_suites
    from issue.readme import write as write_readme
    from issue.runner import (capture_logs, check_hosts, docker_cleanup, docker_down,
                              docker_up, nodes_for, run_single, wait_nodes_ready)
    from issue.suites import ALL_MODELS, ALL_SUITES, build_runs, run_key

    suites = args.suite or ALL_SUITES
    models = args.model or ALL_MODELS
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    is_full = args.suite is None and args.model is None and args.strategy is None

    if args.plots_only:
        history = R.load_history()
        plot_suites(suites, list(history.values()), ts)
        write_readme(history, R.load_run_meta())
        R.write_csv(history.values(), R.RESULTS_DIR / "summary.csv")
        print("Rebuilt plots and README from history.")
        return

    runs = build_runs(suites, models)
    if args.strategy:
        runs = [r for r in runs if r["strategy"] in set(args.strategy)]
    if not runs:
        sys.exit("No runs for the given suites/models/strategies.")

    check_hosts(max(nodes_for(r) for r in runs))
    docker_cleanup()

    out_path = R.RESULTS_DIR / f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.jsonl"
    new_results = []
    done = 0

    # Group by node count; within a group, all-reduce runs reuse the same
    # containers. A reset (down+up) happens only when the previous run left
    # server state (parameter_server / strategy_switch) or the strategy changed.
    groups = defaultdict(list)
    for r in runs:
        groups[nodes_for(r)].append(r)
    for g in groups.values():
        g.sort(key=lambda r: (r["strategy"], r["model"], r["workers"]))

    print(f"Running {len(runs)} runs across suites={suites} models={models}\n")
    t_start = time.perf_counter()
    for n, group in sorted(groups.items()):
        up = False
        prev_strategy = None
        try:
            for run in group:
                done += 1
                needs_reset = (not up
                               or prev_strategy in ("parameter_server", "strategy_switch")
                               or prev_strategy != run["strategy"]
                               or not wait_nodes_ready(n, timeout=1.0))
                print(f"[{done}/{len(runs)}] {run['suite']} · {run['model']} · "
                      f"{run['strategy']} · {n} nodes{'  (reuse)' if not needs_reset else ''}")
                try:
                    if needs_reset:
                        if up:
                            docker_down()
                        docker_up(n, rebuild=args.rebuild)
                        up = True
                        if not wait_nodes_ready(n):
                            raise RuntimeError(f"nodes not ready ({n})")
                    result = run_single(run)
                    if result.get("error"):
                        label = result.get("run_key", f"{run['suite']}_{n}n")
                        capture_logs(n, label.replace("|", "_").replace("/", "-"))
                except Exception as e:
                    result = {"run_key": run_key(run), "suite": run["suite"], "model": run["model"],
                              "strategy": run["strategy"], "workers": run["workers"],
                              "servers": run["servers"], "error": str(e)}
                    up = False

                prev_strategy = run["strategy"]
                R.save_result(result, out_path)
                new_results.append(result)

                if result.get("error"):
                    print(f"  ERROR: {result['error'].splitlines()[-1]}")
                else:
                    acc = result.get("accuracy")
                    acc_s = f" acc={acc:.3f}" if acc is not None else ""
                    print(f"  ok: {result.get('epochs_ran')} epochs in "
                          f"{result.get('train_seconds', 0):.1f}s{acc_s}")
        finally:
            if up and not args.keep_containers:
                try:
                    docker_down()
                except Exception:
                    pass

    total_seconds = time.perf_counter() - t_start
    if is_full:
        R.save_run_meta(total_seconds, ts)

    merged = R.merge(R.load_history(), new_results)
    plot_suites(suites, list(merged.values()), ts)
    write_readme(merged, R.load_run_meta())
    R.write_csv(merged.values(), R.RESULTS_DIR / "summary.csv")
    print(f"\nDone in {total_seconds:.0f}s. Results → {out_path.relative_to(REPO_ROOT)}; "
          "plots and README updated.")


if __name__ == "__main__":
    main()
