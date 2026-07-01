"""Derived throughput/convergence metrics for a finished run.

- epochs_per_sec:   epochs processed per second (throughput, but biased by batch).
- samples_per_sec:  samples processed per second (throughput, batch-invariant —
                    the fair cross-batch metric).
- loss_per_sec:     loss reduction (first - final) per second (convergence speed).
- accuracy_per_sec: final accuracy reached per second (convergence speed).
"""


def derive(result):
    secs = result.get("train_seconds")
    losses = result.get("loss_history") or []
    samples = result.get("train_samples")

    result["epochs_ran"] = len(losses)
    result["final_loss"] = losses[-1] if losses else None
    result["epochs_per_sec"] = (len(losses) / secs) if secs else None

    if samples and secs:
        result["samples_per_sec"] = (samples * len(losses)) / secs
    else:
        result["samples_per_sec"] = None

    if losses and secs:
        result["loss_per_sec"] = (losses[0] - losses[-1]) / secs
    else:
        result["loss_per_sec"] = None

    acc = result.get("accuracy")
    result["accuracy_per_sec"] = (acc / secs) if (acc is not None and secs) else None
    return result
