# Pulse benchmark runs: JunoBench and Deep4ge

Part of [how_to_stream_Keras](../README.md): the same interest in what a
training run exposes about itself, applied to debugging agents.

Harness, raw results and findings from running [Pulse](https://github.com/codeyash09/PulseML)
unattended against two public benchmarks, with Claude Code (Opus 5) as a
reference point on the identical instances.

Two questions were being asked:

1. Can Pulse, given a cheap model over OpenRouter, find and fix real bugs
   without a person at the keyboard?
2. Where does it lose the bug between diagnosing it and writing the fix?

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
| Pulse + DeepSeek V4 Flash, after the fixes | **6/8** |
| Pulse + DeepSeek V4.1 Flash, before the fixes | 2/8 |
| Pulse + DeepSeek V4 Flash, before the fixes | 1/8, then 0/8 on a repeat run |

The fixes are in `codeyash09/PulseML@d9b7668`; nothing about the benchmark,
the model or the prompts changed between the before and after runs.

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
| Pulse + DeepSeek V4.1 Flash | **4/8** |
| Pulse + DeepSeek V4 Flash | **4/8** |

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

**Pulse's losses here are not missed diagnoses.** In three of V4.1's four
failures it identified the fault correctly and then broke the program with
unrelated edits in the same patch:

- `sumsign_init`: replaced `zeros` with `glorot_uniform` (correct), then
  deleted the `model.fit` call and changed the logging callback's signature, so
  training never ran.
- `multilabel_loss`: restored the right loss, then added early stopping and
  rewrote label encoding, ending below the correct program's accuracy.
- `multilabel_init`: removed the fault, then rewrote label handling and
  shuffling, losing ~10% accuracy.

Claude Code changed only the faulty line in 7 of 8 (the eighth set dropout to
0.2 where the original was 0.1).

Pulse's live detection does work on these faults: on a planted
`Adam(learning_rate=5.0)`, its detector fired mid-training
(*"'val_loss' has consistently worsened"*), it diagnosed the learning rate as
"~5000× Adam's default step size", fixed it and restarted — with no question
asked.

## What this surfaced in Pulse

Eleven bugs, all fixed in `codeyash09/PulseML@d9b7668`:

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
results/junobench/   per-run verify_results.json + per-case diffs
results/deep4ge/     per-run score_results.json + diffs, screening report,
                     instances/ (buggy + correct programs, fault metadata)
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
```

Pulse reads its unattended settings from a `pulse_config.json` — see
`pulse_config.example.json`. No key is stored in this repository.
