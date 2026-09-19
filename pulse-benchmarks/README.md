# Pulse benchmark runs: JunoBench, Deep4ge, and detection

Part of [how_to_stream_Keras](../README.md): the same interest in what a
training run exposes about itself, applied to debugging agents.

Harness, raw results and findings from running [Pulse](https://github.com/codeyash09/PulseML)
unattended against two public benchmarks, with Claude Code (Opus 5) as a
reference point on the identical instances.

Three questions were being asked:

1. Can Pulse, given a cheap model over OpenRouter, find and fix real bugs
   without a person at the keyboard?
2. Where does it lose the bug between diagnosing it and writing the fix?
3. Does it notice, on its own, that a run has gone wrong — and does it stay
   quiet when the run is fine?

## Results

### JunoBench — 8 notebook crashes

[JunoBench](https://huggingface.co/datasets/PELAB-LiU/JunoBench) instances
`NBspecific_1`..`NBspecific_8`: real Kaggle notebooks that crash because cells
ran out of order. A case passes when the patched script, re-run **without**
Pulse, finishes cleanly.

| Tool | Score |
|---|---:|
| Claude Code (Opus 5) | **8/8** |
| Pulse + DeepSeek V4.1 Flash, after the fixes | **8/8** |
| Pulse + DeepSeek V4 Flash, after the fixes | **6/8** (4/8 after round 2, see below) |
| Pulse + DeepSeek V4.1 Flash, before the fixes | 2/8 |
| Pulse + DeepSeek V4 Flash, before the fixes | 1/8, then 0/8 on a repeat run |

The fixes are in `codeyash09/PulseML@d9b7668` (round 1) and `@7798e22`
(round 2); nothing about the benchmark, the model or the prompts changed
between the before and after runs.

Case by case, after the fixes:

| Case | Crash | V4 | V4.1 | Claude Code |
|---|---|:--:|:--:|:--:|
| NBspecific_1 | `tf_idf` not defined | ✅ | ✅ | ✅ |
| NBspecific_2 | `X_train` not defined | ✅ | ✅ | ✅ |
| NBspecific_3 | shapes (68,) vs (272,) | ✅ | ✅ | ✅ |
| NBspecific_4 | `sns` not defined | ❌ | ✅ | ✅ |
| NBspecific_5 | column dropped twice | ❌ | ✅ | ✅ |
| NBspecific_6 | `history` not defined | ✅ | ✅ | ✅ |
| NBspecific_7 | `'%percent emission'` missing | ⚠️ | ⚠️ | ⚠️ |
| NBspecific_8 | column dropped twice | ✅ | ✅ | ✅ |

⚠️ NBspecific_7: the cell that creates the missing column sits *after* the
crash, so it is not in the notebook view any tool sees. All three sorted by an
existing column instead — the same workaround, counted as a pass for running
clean but not as the reference fix.

### Deep4ge — 8 injected training faults

[Deep4ge](https://github.com/SigmaJahan/deep4ge) faults don't crash anything;
the model just trains worse. A case passes when the patched program trains back
to the correct program's level (or is an exact revert of the fault).

| Tool | Score |
|---|---:|
| Claude Code (Opus 5) | **8/8** (7 exact reverts) |
| Pulse + DeepSeek V4.1 Flash, after round 2 | **8/8** |
| Pulse + DeepSeek V4.1 Flash, before round 2 | 4/8 |
| Pulse + DeepSeek V4 Flash, before round 2 | 4/8 |

### What Pulse noticed by itself

Deep4ge faults don't crash, so a case can be caught two ways: Pulse's own
monitoring flags the run while it trains, or nothing flags it and the fault is
only found when the agent is asked afterwards. Both count as a fix; only the
first is Pulse doing what it exists to do. After round 2, on V4.1 Flash:

| Instance | Fault | Caught by | What fired |
|---|---|---|---|
| trivial_init | initializer → `zeros` | ✅ live detection | *"'loss' is no better at the end of the run than at the start (0.6932 → 0.6932 over 10 epochs), and 'accuracy' never moved from 0.4968"* |
| sumsign_init | initializer → `zeros` | ✅ live detection | *"'loss' is no better at the end of the run than at the start (0.6932 → 0.6933 over 10 epochs)"* |
| multilabel_loss | loss → `categorical_hinge` | ✅ live detection | *"'loss' is no better at the end of the run than at the start (0.5135 → 0.5049 over 10 epochs)"* |
| sumsign_lr | learning rate → 0.603 | ✅ live detection | train/validation divergence: *"train_loss 7761 → 0.6992, while val_loss 0.7038 → 1.492e+04"* |
| linear_init | initializer → `zeros` | ⚠️ static check before training | the start-of-run ML lint, not the runtime detectors |
| multilabel_init | initializer → `zeros` | ❌ not detected | fixed only once the agent was asked |
| sumsign_dropout | dropout → 1.0 | ❌ not detected | fixed only once the agent was asked |
| trivial_dropout | dropout → 1.0 | ❌ not detected | fixed only once the agent was asked |

**4 of 8 detected while training, 1 caught statically before training, 3 not
detected at all.** Three of the four live detections come from a check added in
round 2 (a run whose loss ends no better than it started); before it, the
detectors fired **zero times across 16 runs**, because they read step-level
history while Keras metrics arrive per epoch in a different store.

The three undetected ones are honest misses: a dropout of 1.0 and a zeroed
initializer deeper in the network still produce a loss curve that improves,
and Pulse has no way to know what accuracy was achievable.

| Instance | Fault | V4 | V4.1 | Claude Code |
|---|---|:--:|:--:|:--:|
| sumsign_lr | learning rate 0.001 → 0.603 | ✅ | ✅ | ✅ |
| trivial_dropout | dropout 0.1 → 1.0 | ✅ | ✅ | ✅ |
| sumsign_init | initializer → `zeros` | ✅ | ❌ | ✅ |
| linear_init | initializer → `zeros` | ✅¹ | ❌ | ✅ |
| sumsign_dropout | dropout 0.2 → 1.0 | ❌ | ✅ | ✅ |
| trivial_init | initializer → `zeros` | ❌ | ✅ | ✅ |
| multilabel_init | initializer → `zeros` | ❌ | ❌ | ✅ |
| multilabel_loss | loss → `categorical_hinge` | ❌ | ❌ | ✅ |

¹ Counted as a pass — the fault was removed — but the same patch also
normalized the target values, which drives the loss toward zero regardless, so
the measurement flatters it.

**Before round 2, Pulse's losses were not missed diagnoses.** In three of
V4.1's four failures it identified the fault correctly and then broke the
program with unrelated edits in the same patch:

- `sumsign_init`: replaced `zeros` with `glorot_uniform` (correct), then
  deleted the `model.fit` call and changed the logging callback's signature, so
  training never ran.
- `multilabel_loss`: restored the right loss, then added early stopping and
  rewrote label encoding, ending below the correct program's accuracy.
- `multilabel_init`: removed the fault, then rewrote label handling and
  shuffling, losing ~10% accuracy.

Claude Code changed only the faulty line in 7 of 8 (the eighth set dropout to
0.2 where the original was 0.1).

Round 2 addressed exactly that: the fix prompt now states that the job is to fix
a bug in a training run and nothing else, and a fix its own verification
rejected is no longer applied anyway. All three of those cases now pass.

Pulse's live detection does work on these faults: on a planted
`Adam(learning_rate=5.0)`, its detector fired mid-training
(*"'val_loss' has consistently worsened"*), it diagnosed the learning rate as
"~5000× Adam's default step size", fixed it and restarted — with no question
asked.

### Detection — 67 labelled runs

The two benchmarks above measure fixing. This one measures noticing, which is the part
Pulse exists for, and it has two halves that are equally load-bearing: catching runs
that are broken, and leaving runs that are fine alone. A detector that interrupts a
healthy run gets switched off by its user, and then it detects nothing at all — so
false alarms are scored here as failures, exactly like misses.

67 labelled training runs, replayed epoch by epoch into a fresh detector. 39 broken:
NaN and inf mid-run, exploding loss, a slow divergence, a divergence after thirty good
epochs, a frozen loss, vanishing and zero gradients, a learning rate jump, a schedule
that decays to zero, overfitting, validation regression, oscillation, oscillation with
growing amplitude, accuracy stuck at chance, a leak visible in epoch 1 and a leak that
takes ten epochs to saturate, NaN weights, a crawl, a sustained spike, a metric frozen
while the loss moves, an exhausted data iterator repeating the same readings, per-epoch
state resetting, a model collapsing to the majority class, a loss improving while
accuracy sits at chance, an unbounded weight norm, a step time that climbs all run, an
ELBO parked at −3.2, a negative loss going the wrong way. 25 healthy: smooth, noisy,
converged, warmup, a deliberate LR schedule, a cyclical one, SGDR warm restarts,
mixed-precision loss scaling (values ×65536), curriculum learning, multi-task losses on
four different scales, reinforcement learning, a GAN, fine-tuning, validation better
than train, an imbalanced task where accuracy starts at 0.95, step-level readings,
per-epoch spikes at each reshuffle, custom metric names, gaps in the history, 1e-7 and
5e6 values, and a short early-stopped run. Plus 3 judgement calls, reported either way.

| Detector | Caught | False alarms |
|---|---:|---:|
| The engine (`pulse_detect`), after this work | **39/39** | **0/25** |
| What `pulse run` actually used, before | 36/39 | **10/25** |

Pulse had *three* deterministic detectors. `pulse_detect.DetectionEngine` is the one its
tests cover — and it was reachable only under `PULSE_MODE=stream`. A normal `pulse run`
used a second, untested implementation, and the Tk dashboard a third. All three paths
now run the engine (`codeyash09/PulseML@492b56e`); `PULSE_LEGACY_DETECTOR=1` restores
the old one, which is how the second row above is still reproducible.

Thresholds that are right and thresholds fitted to one lucky curve score the same on
one draw, so every curve's noise was redrawn 30 times at five sensitivities — 10,050
replays:

| sensitivity | caught | false alarms |
|---|---|---|
| 0.1 | 1170/1170 (100%) | 3/750 (0.4%) |
| 0.3 *(default)* | 1170/1170 (100%) | 4/750 (0.5%) |
| 0.5 | 1170/1170 (100%) | 9/750 (1.2%) |
| 0.7 | 1170/1170 (100%) | 26/750 (3.5%) |
| 0.9 | 1170/1170 (100%) | 110/750 (14.7%) |

Recall holds at 100% across the dial while false alarms rise monotonically: the dial
buys quiet, not detection. Median latency is 16 epochs to the first raise.

Two more suites, because a detector's failure modes are not all statistical:

- **63 hostile inputs** — `None` and strings in the history, NaN from the first reading,
  1e300, 200 variables, unicode names, a 50k-step history, a history that shrinks, a
  variable that disappears mid-run, four ranks reporting the same metric, steps that go
  backwards, an agent-supplied baseline of NaN, a thousand identical updates, and every
  dtype a training loop reports. Four of these were real crashes or hangs when first
  run, and a detector that raises takes the training run down with it.
- **8 wiring checks** — Keras hands loss and `val_loss` to a callback's `logs` dict,
  never as locals. Sixteen Deep4ge runs detected nothing for exactly this reason, with
  every check working correctly. These drive the real ingestion function with the log
  dicts Keras emits and then ask the real detector.

## What this surfaced in Pulse

### Round 4 (`codeyash09/PulseML@492b56e`, `@180111a`) -- widening the net

Going from 37 labelled runs to 67 found six failure modes that were missed outright and
two healthy runs that were flagged.

| Problem | Effect |
|---|---|
| float32 readings were discarded entirely | `isinstance(v, (int, float))` is true for numpy's float64, which subclasses float, and false for float32 — the Keras default, and everything under mixed precision. Every reading of such a run was dropped, so **no check ran at all**, including the NaN check. Torch scalars, 0-dim arrays and Decimals went the same way |
| Nothing noticed the same readings coming round again | An exhausted iterator being re-used, or per-epoch state resetting: bit-for-bit repetition that cannot happen by chance |
| Nothing noticed a run getting slower | A step time climbing all run is a leak that ends as an out-of-memory kill hours later. It is not a loss, a score, a norm or a learning rate, so nothing looked at it |
| A steadily growing weight norm never tripped the spike test | The recent average it was compared against grew with it |
| A leak that took ten epochs to saturate was invisible | "Suspiciously perfect" only looked at a metric's first six readings |
| A loss improving while accuracy sat at chance was INFO | Shuffled labels. The loss is optimising something the metric does not measure, and the fix is never in the optimiser |
| Every ratio test changed direction with the sign | An ELBO or a log-likelihood sitting flat at −3.2 forever passed `end >= start * 0.98` trivially |
| A slope alone flagged anything that oscillates | Over less than one period a sine is a straight line: an RL policy loss measured 6 standard errors and explained 75% of its own variance while going nowhere. Real divergences measured 16 to 59, explaining over 90% |
| A quantised metric read as frozen | 197 right out of 200 is exactly 0.985 every epoch. That is the score having converged, not the run having died |
| A third detector in the Tk dashboard | Migrated to the engine with the other two |

And the one bug Pulse could not see from inside a training script at all: a **syntax
error in the script itself**. Python compiles the whole file before `auto_track()` on
line 2 ever runs. `pulse run` had a repair path for it that did not work — applying the
fix handed control to the restart machinery, which re-ran a temporary copy and skipped
the rest, so the user's file stayed broken with the fix stranded in a temp file; a rate
limit was reported as a failed repair; and with no terminal it stopped at "setup was not
completed". Fixed, and verified end to end: the file is now repaired, written back, and
compiles on its own.

### Round 3 (`codeyash09/PulseML@d682240`) -- detection

| Problem | Effect |
|---|---|
| Two detectors, and the tested one was not the one running | `pulse run` used `PulseCLI._check_for_trouble`; the benchmarked engine needed `PULSE_MODE=stream`. 7 of 15 healthy runs interrupted (10 of 25 on the wider set) |
| No check for a loss that climbs | A run diverging steadily without ever spiking was caught 15 times in 30 — by luck, through other checks |
| "This run is not learning" was a coin flip on a noisy curve | It disqualified itself if a single reading had ever dipped below the opening, which noise does |
| Oscillation was a white-noise detector | It asked for ≥12 direction changes in 19 readings; iid noise flips about two in three |
| Stagnation could not tell converged from stalled | A run that came down 99% and sat at its floor read the same as one that stopped at 30% |
| Overfitting and drift compared two numbers, not two distributions | On a small validation split that is mostly noise: healthy peaks at 2.0σ, real faults reach 6.6σ |
| Per-variable checks only looked at `tracked_vars` | A metric static discovery missed was never checked at all — the Deep4ge failure, in its general form |
| Info-level findings paused training | "accuracy has not moved" on a fine-tune holding 97% called an agent |

### Round 2 (`codeyash09/PulseML@7798e22`) -- Deep4ge 4/8 to 8/8

| Problem | Effect |
|---|---|
| Detectors read step-level history only | Keras metrics arrive per epoch in `epoch_scalar_histories`, so the detectors inspected an empty history: **zero firings across 16 runs** while 50 epochs of loss and accuracy sat beside them |
| No check for a run that never learns | Every other check looks for training going *wrong*; a loss flat from the first epoch to the last tripped none of them |
| A rejected fix was applied as "best effort" | After a correct fix had landed, an edit its own verification had rejected deleted the `model.fit` call |
| Nothing said the job was "fix the bug only" | Correct diagnoses arrived wrapped in unrelated edits -- early stopping, rewritten label encoding, layers removed -- which is what actually failed the cases |
| Asking to see more code ended the pipeline | The fix pass accepted nothing but finished JSON, so *"I need to see lines 295-305"* killed four of eight JunoBench cases in one run |
| Snippet matching ignored whitespace only around lines | Replaying every rejected patch showed one in five would land under whitespace-insensitive matching, with no model call |
| Replaced code left behind under a banner | `=====Pulse Change====` blocks nested with every fix; patches then tried to repair Pulse's own leftovers |
| A fix inside an envelope was discarded | `{"pulse_analysis": ..., "json": {old/new/...}}` was rejected whole instead of read one level down |

### Round 1 (`codeyash09/PulseML@d9b7668`) -- JunoBench 1/8 to 8/8

Eleven bugs:

| Problem | Effect |
|---|---|
| Per-pass caps of 200/300/700 output tokens | A reasoning model spent the budget before emitting text: empty reply, pipeline aborted |
| The fix pass never received the diagnosis | Each pass re-derived it from the code and traceback alone |
| Crash-time retries queued on a daemon thread | The crashed script exits first, so the retry never ran |
| A failed verify/sweep discarded a fix already on disk | Correct fixes thrown away, restart skipped |
| No crash hook when no trackable variables exist | Scripts failing on their first lines got no agent at all |
| Variables reported as "not run yet" | The sampler hadn't run yet on fast crashes; the agent diagnosed from false state |
| `_resolve_fix_path` couldn't resolve a path | The "re-quote the snippet exactly" retry silently never ran |
| Duplicated blocks refused as ambiguous | The most common notebook bug couldn't be patched; now the copy at the crash line wins |
| Rejected fixes printed nothing | A skipped fix was indistinguishable from silence |
| Restarts nesting without limit | 347 agent calls in 57 minutes on one case; a second, non-crash path reached 92 |
| OpenRouter limited to two hardcoded free models | Any other model needed the Custom entry and a hand-named env var |

## What this surfaced in the benchmarks

- **Deep4ge's learning-rate operator (HLR) is broken.** `_wrap_lr` builds
  `ast.Name(id="__import__('tensorflow.keras.backend')")` and replaces the
  whole optimizer argument, so the program dies at `compile()` rather than
  training worse. `harness/deep4ge/make_instances.py` implements a correct
  source-level version (`SRC_HLR`).
- **Deep4ge's Weight/Activation/Regularization/Layer operators mutate a built
  Keras model object**, so they change nothing in a source file and cannot
  produce a bug an agent could fix by editing code. The same fault definitions
  are implemented as source edits (`SRC_WCI`, `SRC_ACH`, `SRC_ARM`, `SRC_RCD`,
  `SRC_LCN`), with the same value ranges from the paper's catalog.
- **Many injected faults don't measurably hurt training.** Of 47 screened
  variants, most failed the "consistently and clearly worse" test — consistent
  with Deep4ge's own statistical killing criterion.

## Method

**Environments.** JunoBench: Python 3.10 with the benchmark's pinned versions
(TensorFlow 2.17, scikit-learn 1.2.2, pandas 2.1.4, numpy 1.26.4). Deep4ge:
Python 3.11 + TensorFlow 2.15.1, matching its `requirements.txt`. All 8
JunoBench crashes reproduce exactly and all 8 reference fixes run clean before
any tool is involved (`results/junobench/baseline_results.json`).

**What the agent sees.** JunoBench notebooks are linearized into a script in
real execution order, with never-run cells before the crash kept as commented
code — what a person looking at the notebook sees. JunoBench's own hint
comments ("this crash is because this cell was executed more than once") are
stripped. Run folders contain only the script and its data; fixed notebooks,
labels and other instances are unreadable during a run.

**Scoring.** JunoBench: the patched script is re-run without Pulse and must
exit cleanly; every diff was also read by hand to separate real fixes from
workarounds. Deep4ge: the patched program is retrained three times and compared
against the correct program's own spread, plus an exact-revert check.

**Fairness.** Claude Code got the same scripts, the same prompt, no web access,
no connectors, and only a wrapper that runs Python. Every tool call in its
transcripts was audited afterwards; none touched the answer files.

## Caveats

- This compares *tool + model* pairs, not tools alone: Opus 5 against DeepSeek
  Flash models. It does not isolate Pulse from its model.
- 8 instances per benchmark is small; JunoBench has 111 cases and Deep4ge 59
  seed programs.
- The 8 Deep4ge instances cover 4 fault categories (Weight ×4, Regularization
  ×2, Hyperparameter ×1, Loss ×1) because the other categories' faults did not
  degrade training measurably on the seeds whose correct version learns.
- Deep4ge's heavyweight logging callback is replaced by a metrics-only
  equivalent (`harness/deep4ge/screen_callback.py`). It only observes training,
  but it is a deviation from the published instrumentation.

## Layout

```
harness/junobench/   nb2py.py (notebook -> script), run.py (baseline/pulse/verify),
                     trace/ (logs every model call), cc/ (Claude Code runner)
harness/deep4ge/     make_instances.py (fault injection + screening), run.py,
                     driver.py (asks Pulse's agent after training), cc/
harness/detection/   scenarios.py (67 labelled runs), run_detectbench.py (score),
                     sweep.py (redraw the noise, turn the dial), edge_cases.py
                     (63 hostile inputs), wiring.py (Keras logs reach the detector),
                     legacy.py (the same runs through the default path)
results/junobench/   per-run verify_results.json + per-case diffs
results/deep4ge/     per-run score_results.json + diffs, screening report,
                     instances/ (buggy + correct programs, fault metadata),
                     v41_flash_round2/detection_evidence.txt (what fired, per case)
results/detection/   benchmark_sensitivity_0.3.json (per-run verdicts), sweep.txt,
                     legacy_detector.txt (the default path before the switch),
                     robustness_and_wiring.txt
```

`results/junobench/v4_flash_before_run1/verify_results_all_cases.json` has one
distorted row: NBspecific_6 was re-verified concurrently by two processes; its
isolated result is in the file beside it.

## Reproducing

```bash
cd pulse-benchmarks

# JunoBench (from a directory holding benchmark/ and the harness)
python harness/junobench/run.py baseline      # crashes reproduce, fixes run clean
BENCH_CONFIG=pulse_config.json python harness/junobench/run.py pulse
python harness/junobench/run.py verify

# Deep4ge
python harness/deep4ge/make_instances.py      # inject faults, keep only the ones that hurt
python harness/deep4ge/run.py pulse
python harness/deep4ge/run.py score

# Detection (runs against an installed Pulse; no model, no API key, no GPU)
python harness/detection/run_detectbench.py
python harness/detection/sweep.py --seeds 30
python harness/detection/edge_cases.py
python harness/detection/wiring.py
PULSE_LEGACY_DETECTOR=1 python harness/detection/legacy.py   # the old detector
```

Pulse reads its unattended settings from a `pulse_config.json` — see
`pulse_config.example.json`. No key is stored in this repository.
