"""Labelled training curves for benchmarking Pulse's detection.

Each scenario is a run replayed epoch by epoch into the engine. `expect` is the
check that should raise (None = nothing should raise: these are the false-alarm
tests, and they matter as much as the positives -- a detector that fires on a
healthy run gets turned off).

Curves are deterministic (seeded) so a result is reproducible.
"""
import math

import numpy as np

EPOCHS = 60


# Every curve's noise is seeded so a result is reproducible. Shifting this offset
# redraws all of them, which is how the sweep checks that a threshold was fitted to
# the shape of a run and not to one lucky draw of its noise.
SEED_OFFSET = 0


def set_seed_offset(offset):
    global SEED_OFFSET
    SEED_OFFSET = int(offset)


def _rng(seed):
    return np.random.default_rng(seed + SEED_OFFSET * 1009)


def descending(n=EPOCHS, start=2.0, floor=0.05, noise=0.01, seed=0):
    """A healthy loss: fast early progress, slowing, small noise."""
    r = _rng(seed)
    return [float(floor + (start - floor) * math.exp(-3.0 * i / n) + r.normal(0, noise)) for i in range(n)]


def rising(n=EPOCHS, start=0.5, rate=0.03, seed=1):
    r = _rng(seed)
    return [float(start * (1 + rate) ** i + r.normal(0, 0.01)) for i in range(n)]


def flat(n=EPOCHS, value=0.693, noise=0.0005, seed=2):
    r = _rng(seed)
    return [float(value + r.normal(0, noise)) for i in range(n)]


def accuracy_from(loss, chance=0.5, best=0.97):
    lo, hi = min(loss), max(loss)
    span = max(hi - lo, 1e-9)
    return [float(chance + (best - chance) * (hi - v) / span) for v in loss]


# --------------------------------------------------------------------------- faults


def nan_midrun():
    loss = descending()
    loss[25:] = [float("nan")] * (len(loss) - 25)
    return {"loss": loss}, None


def inf_midrun():
    loss = descending()
    loss[30:] = [float("inf")] * (len(loss) - 30)
    return {"loss": loss}, None


def exploding_loss():
    loss = descending()[:20] + [float(0.3 * (2.2 ** i)) for i in range(40)]
    return {"loss": loss}, None


def diverging_slowly():
    return {"loss": rising(start=0.6, rate=0.02)}, None


def frozen_from_start():
    """Dropout(1.0), a zeroed initializer: the loss never moves at all."""
    loss = flat()
    return {"loss": loss, "accuracy": [0.5] * EPOCHS}, None


def learns_then_stops():
    """Real progress for 15 epochs, then nothing for 45 at a bad value."""
    head = descending(n=15, start=2.0, floor=1.2)
    return {"loss": head + flat(n=45, value=1.2, noise=0.001, seed=7)}, None


def gradient_explosion():
    return {"loss": descending(), "grad_norm": [float(1.0 * (1.6 ** i)) for i in range(EPOCHS)]}, None


def gradient_vanishing():
    return {"loss": flat(value=0.69),
            "grad_norm": [float(max(1e-12, 0.5 * (0.5 ** i))) for i in range(EPOCHS)]}, None


def lr_jumped_up():
    lr = [0.001] * 30 + [0.5] * 30
    return {"loss": descending()[:30] + rising(n=30, start=0.4, rate=0.05), "lr": lr}, None


def overfitting_classic():
    train = descending(start=1.5, floor=0.01)
    val = [float(0.6 + 0.02 * max(0, i - 20)) for i in range(EPOCHS)]
    return {"loss": train, "val_loss": val}, None


def validation_regression():
    val = [float(0.5 + 0.01 * i) for i in range(EPOCHS)]
    return {"loss": descending(), "val_loss": val}, None


def oscillating_loss():
    r = _rng(11)
    return {"loss": [float(1.0 + 0.6 * math.sin(i * 2.4) + r.normal(0, 0.05)) for i in range(EPOCHS)]}, None


def accuracy_at_chance():
    return {"loss": flat(value=0.693), "accuracy": [0.5] * EPOCHS, "val_accuracy": [0.502] * EPOCHS}, None


def perfect_validation():
    """A leak: validation is perfect from the first epoch."""
    return {"loss": descending(), "val_accuracy": [1.0] * EPOCHS, "accuracy": accuracy_from(descending())}, None


def nan_weights():
    return {"loss": descending()}, {"dense/kernel": {"nan": 12, "inf": 0, "shape": [64, 32]}}


def slow_crawl():
    """Technically improving, far slower than it should: 2.0 -> 1.94 in 60 epochs."""
    return {"loss": [float(2.0 - 0.001 * i) for i in range(EPOCHS)]}, None


def loss_spike_sustained():
    """A spike that does not recover -- divergence, not one bad batch."""
    return {"loss": descending()[:25] + [float(9.0 + 0.2 * i) for i in range(35)]}, None


def metric_frozen_while_loss_moves():
    return {"loss": descending(), "accuracy": [0.62] * EPOCHS}, None


# --------------------------------------------------------------------------- healthy


def healthy_smooth():
    loss = descending()
    return {"loss": loss, "val_loss": descending(seed=3, floor=0.08),
            "accuracy": accuracy_from(loss), "val_accuracy": accuracy_from(loss, best=0.95)}, None


def healthy_noisy():
    """Small dataset: the same descent with a lot of noise on it."""
    loss = descending(noise=0.12, seed=4)
    return {"loss": loss, "val_loss": descending(noise=0.18, seed=5, floor=0.1)}, None


def healthy_converged_flat():
    """Descends, then sits at a good value for 40 epochs. The classic false alarm."""
    return {"loss": descending(n=20, start=2.0, floor=0.02) + flat(n=40, value=0.02, noise=0.001, seed=6)}, None


def healthy_warmup():
    """Learning-rate warmup: the loss rises for the first few epochs by design."""
    return {"loss": [0.9, 1.1, 1.3, 1.25] + descending(n=56, start=1.2, floor=0.05)}, None


def healthy_lr_schedule():
    """A step schedule drops the learning rate 10x at epoch 30 -- deliberate."""
    lr = [0.01] * 30 + [0.001] * 30
    return {"loss": descending(), "lr": lr}, None


def healthy_transient_spike():
    """One bad batch at epoch 30, recovered by the next epoch."""
    loss = descending()
    loss[30] = loss[30] * 6
    return {"loss": loss}, None


def healthy_finetune():
    """Fine-tuning: already good, improving by tiny amounts."""
    return {"loss": [float(0.050 - 0.00015 * i) for i in range(EPOCHS)],
            "accuracy": [float(0.970 + 0.0002 * i) for i in range(EPOCHS)]}, None


def healthy_short_run():
    loss = descending(n=8)
    return {"loss": loss, "val_loss": descending(n=8, seed=8, floor=0.1)}, None


def healthy_noisy_validation():
    """Validation on a small split bounces around while training improves."""
    r = _rng(9)
    val = [float(0.4 + r.normal(0, 0.12)) for _ in range(EPOCHS)]
    return {"loss": descending(), "val_loss": val}, None


def healthy_gan_oscillation():
    """Adversarial training: the generator loss legitimately oscillates."""
    r = _rng(10)
    return {"g_loss": [float(1.2 + 0.35 * math.sin(i / 2.0) + r.normal(0, 0.05)) for i in range(EPOCHS)],
            "d_loss": [float(0.7 + 0.2 * math.cos(i / 2.0) + r.normal(0, 0.05)) for i in range(EPOCHS)]}, None


def healthy_late_breakthrough():
    """Flat for 30 epochs, then learns (grokking). Flagging the flat part is
    defensible, so this is scored separately rather than counted as a failure."""
    return {"loss": flat(n=30, value=0.69, noise=0.002) + descending(n=30, start=0.69, floor=0.05)}, None


def healthy_perfect_easy_task():
    """A genuinely easy task: accuracy really is 1.0. Pulse warns here by
    design ('suspiciously perfect'), so this is scored separately too."""
    return {"loss": descending(floor=0.001), "val_accuracy": [1.0] * EPOCHS}, None


def healthy_step_level():
    """Per-batch values rather than per-epoch: noisier, many more points."""
    r = _rng(12)
    return {"loss": [float(2.0 * math.exp(-3.0 * i / 600) + abs(r.normal(0, 0.25))) for i in range(600)]}, None


def healthy_custom_names():
    """Nothing is called 'loss': does the engine still recognise the run?"""
    loss = descending()
    return {"train_objective": loss, "eval_objective": descending(seed=13, floor=0.09),
            "dice": accuracy_from(loss, chance=0.3, best=0.9)}, None


def healthy_missing_values():
    """Gaps in the history: a metric only evaluated every few epochs."""
    loss = descending()
    val = [descending(seed=14, floor=0.08)[i] if i % 5 == 0 else None for i in range(EPOCHS)]
    return {"loss": loss, "val_loss": val}, None


def healthy_tiny_values():
    return {"loss": [float(1e-7 * math.exp(-i / 20)) for i in range(EPOCHS)]}, None


def healthy_large_values():
    return {"loss": [float(5e6 * math.exp(-3.0 * i / EPOCHS)) for i in range(EPOCHS)]}, None


# name, builder, expected check (None = nothing should raise), tags
SCENARIOS = [
    # ---- must fire
    ("nan_midrun", nan_midrun, "nonfinite", "fault"),
    ("inf_midrun", inf_midrun, "nonfinite", "fault"),
    ("exploding_loss", exploding_loss, "loss_spike", "fault"),
    ("diverging_slowly", diverging_slowly, "any", "fault"),
    ("frozen_from_start", frozen_from_start, "any", "fault"),
    ("learns_then_stops", learns_then_stops, "any", "fault"),
    ("gradient_explosion", gradient_explosion, "norm_explosion", "fault"),
    ("gradient_vanishing", gradient_vanishing, "any", "fault"),
    ("lr_jumped_up", lr_jumped_up, "any", "fault"),
    ("overfitting_classic", overfitting_classic, "any", "fault"),
    ("validation_regression", validation_regression, "any", "fault"),
    ("oscillating_loss", oscillating_loss, "any", "fault"),
    ("accuracy_at_chance", accuracy_at_chance, "any", "fault"),
    ("perfect_validation", perfect_validation, "suspiciously_perfect", "fault"),
    ("nan_weights", nan_weights, "tensor_nonfinite", "fault"),
    ("slow_crawl", slow_crawl, "any", "fault"),
    ("loss_spike_sustained", loss_spike_sustained, "any", "fault"),
    ("metric_frozen_while_loss_moves", metric_frozen_while_loss_moves, "any", "fault"),
    # ---- must stay quiet
    ("healthy_smooth", healthy_smooth, None, "healthy"),
    ("healthy_noisy", healthy_noisy, None, "healthy"),
    ("healthy_converged_flat", healthy_converged_flat, None, "healthy"),
    ("healthy_warmup", healthy_warmup, None, "healthy"),
    ("healthy_lr_schedule", healthy_lr_schedule, None, "healthy"),
    ("healthy_transient_spike", healthy_transient_spike, None, "healthy"),
    ("healthy_finetune", healthy_finetune, None, "healthy"),
    ("healthy_short_run", healthy_short_run, None, "healthy"),
    ("healthy_noisy_validation", healthy_noisy_validation, None, "healthy"),
    ("healthy_gan_oscillation", healthy_gan_oscillation, None, "healthy"),
    ("healthy_step_level", healthy_step_level, None, "healthy"),
    ("healthy_custom_names", healthy_custom_names, None, "healthy"),
    ("healthy_missing_values", healthy_missing_values, None, "healthy"),
    ("healthy_tiny_values", healthy_tiny_values, None, "healthy"),
    ("healthy_large_values", healthy_large_values, None, "healthy"),
    # ---- judgement calls: reported, not counted as failures either way
    ("healthy_late_breakthrough", healthy_late_breakthrough, None, "debatable"),
    ("healthy_perfect_easy_task", healthy_perfect_easy_task, None, "debatable"),
]
