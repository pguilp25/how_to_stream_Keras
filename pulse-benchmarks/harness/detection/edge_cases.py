"""Can the detector be crashed, hung, or confused by the data a real run hands it?

A detector that raises an exception takes the training run down with it, which is
strictly worse than not detecting anything. Every case here is something a real
callback has produced: a metric that is None until the first validation pass, a
history of one point, a loss that is legitimately negative, names with unicode in
them, 50k step-level readings, a tensor summary with missing keys.

Nothing here asserts a particular finding: the contract is that the engine keeps
running, keeps its promises about types, and stays fast.

    python3 edge_cases.py
"""
import math
import os
import sys
import time

# Point PULSE_SRC at a Pulse checkout, or rely on an installed pulse package.
sys.path.insert(0, os.environ.get("PULSE_SRC")
                or os.path.expanduser("~/pulseml/pulse-pkg/src"))

from pulse import pulse_detect as detect  # noqa: E402

PASS, FAIL = [], []


def check(label, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - that is the point of the test
        FAIL.append("%s -> %s: %s" % (label, type(exc).__name__, exc))
    else:
        PASS.append(label)


def engine():
    return detect.DetectionEngine(sensitivity=0.3, confirmations=2)


def feed(histories, **kw):
    """One update must always return the two lists, whatever it was given."""
    out = engine().update(histories, **kw)
    assert isinstance(out, dict) and "raised" in out and "cleared" in out, out
    assert isinstance(out["raised"], list) and isinstance(out["cleared"], list)
    for finding in out["raised"]:
        assert finding.severity in (detect.CRITICAL, detect.WARNING, detect.INFO)
        assert isinstance(finding.message, str) and finding.message
        assert 0.0 <= finding.confidence <= 1.0, finding.confidence
    return out


# ------------------------------------------------------------------ empty and tiny
check("no histories at all", lambda: feed({}))
check("None instead of histories", lambda: feed(None))
check("a variable with an empty history", lambda: feed({"loss": []}))
check("a single reading", lambda: feed({"loss": [1.0]}))
check("two readings", lambda: feed({"loss": [1.0, 0.9]}))
check("every value identical", lambda: feed({"loss": [0.5] * 100}))
check("all zeros", lambda: feed({"loss": [0.0] * 100}))

# ------------------------------------------------------------------ wrong types
check("None values in the history", lambda: feed({"loss": [1.0, None, 0.8, None, 0.6] * 8}))
check("all values None", lambda: feed({"loss": [None] * 40}))
check("strings in the history", lambda: feed({"loss": ["0.5", "oops", 0.4] * 10}))
check("booleans in the history", lambda: feed({"acc": [True, False, True] * 10}))
check("a nested list", lambda: feed({"loss": [[1.0], [0.9], 0.8] * 10}))
check("a dict as a history", lambda: feed({"loss": {"a": 1}}))
check("a numpy-ish object", lambda: feed({"loss": [type("F", (), {"__float__": lambda s: 1.0})()] * 12}))

# ------------------------------------------------------------------ hostile numbers
check("negative loss throughout", lambda: feed({"loss": [-1.0 - 0.01 * i for i in range(60)]}))
check("loss crossing zero", lambda: feed({"loss": [1.0 - 0.05 * i for i in range(60)]}))
check("denormal magnitudes", lambda: feed({"loss": [1e-300 * (0.9 ** i) for i in range(40)]}))
check("enormous magnitudes", lambda: feed({"loss": [1e300 * (0.9 ** i) for i in range(40)]}))
check("nan from the first reading", lambda: feed({"loss": [float("nan")] * 30}))
check("inf and -inf mixed", lambda: feed({"loss": [float("inf"), float("-inf")] * 20}))
check("nan then recovery", lambda: feed({"loss": [1.0, float("nan"), 0.9, 0.8] * 10}))

# ------------------------------------------------------------------ shapes and names
check("histories of different lengths",
      lambda: feed({"loss": [1.0] * 60, "val_loss": [0.9] * 3, "acc": [0.5] * 31}))
check("empty variable name", lambda: feed({"": [1.0] * 20}))
check("unicode names", lambda: feed({"perte_d'entraînement": [1.0] * 20, "précision": [0.5] * 20}))
check("a name that is every hint at once", lambda: feed({"val_loss_accuracy_lr_grad_norm": [1.0] * 20}))
check("two hundred variables",
      lambda: feed({("m%d" % i): [1.0 - 0.001 * j for j in range(30)] for i in range(200)}))

# ------------------------------------------------------------------ steps
check("no step given", lambda: feed({"loss": [1.0] * 20}))
check("negative step", lambda: feed({"loss": [1.0] * 20}, step=-5))
check("huge step", lambda: feed({"loss": [1.0] * 20}, step=2 ** 62))
check("float step", lambda: feed({"loss": [1.0] * 20}, step=3.7))

# ------------------------------------------------------------------ tensor stats
check("tensor stats missing keys", lambda: feed({"loss": [1.0] * 20}, tensor_stats={"w": {}}))
check("tensor stats wrong types",
      lambda: feed({"loss": [1.0] * 20}, tensor_stats={"w": {"nan": "lots", "inf": None}}))
check("tensor stats not a dict", lambda: feed({"loss": [1.0] * 20}, tensor_stats={"w": 5}))
check("tensor stats None", lambda: feed({"loss": [1.0] * 20}, tensor_stats=None))


# ------------------------------------------------------------------ the state machine
def state_machine():
    eng = engine()
    broken = [0.693] * 40                      # frozen: should raise, once confirmed
    raised = []
    for i in range(12, 41):
        raised += eng.update({"loss": broken[:i]}, step=i)["raised"]
    assert raised, "a frozen loss must raise something"
    keys = {f.key for f in raised}
    assert len(raised) == len(keys), "the same finding must not be raised twice while it is active"
    assert eng.current(), "an active finding must show up in current()"
    # it gets fixed: the loss starts moving again
    fixed = broken + [0.693 * (0.9 ** i) for i in range(1, 40)]
    cleared = []
    for i in range(41, len(fixed) + 1):
        cleared += eng.update({"loss": fixed[:i]}, step=i)["cleared"]
    assert cleared, "a finding must clear once the run recovers"


check("raise once, clear on recovery", state_machine)


def immediate_vs_confirmed():
    eng = engine()
    out = eng.update({"loss": [1.0, 0.9, float("nan")]}, step=3)
    assert any(f.check == "nonfinite" for f in out["raised"]), "nan must not wait for a second opinion"
    slow = detect.DetectionEngine(sensitivity=0.3, confirmations=3)
    first = slow.update({"loss": [0.693] * 30}, step=30)["raised"]
    assert not first, "an ordinary finding must wait for its confirmations"


check("nan raises at once, the rest wait", immediate_vs_confirmed)


def idempotent():
    eng = engine()
    hist = {"loss": [0.693] * 40}
    for _ in range(10):
        eng.update(hist, step=40)
    active = eng.current()
    assert len({f.key for f in active}) == len(active), "duplicate findings for the same key"


check("the same data repeated does not pile up findings", idempotent)


def sensitivity_is_monotone():
    """Turning the dial up must never make the engine notice less."""
    curve = {"loss": [1.0 - 0.0005 * i for i in range(60)]}
    seen = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        eng = detect.DetectionEngine(sensitivity=s, confirmations=2)
        fired = set()
        for i in range(5, 61):
            for f in eng.update({"loss": curve["loss"][:i]}, step=i)["raised"]:
                fired.add(f.check)
        seen.append(fired)
    for lower, higher in zip(seen, seen[1:]):
        assert lower <= higher or len(higher) >= len(lower), (lower, higher)


check("higher sensitivity never detects less", sensitivity_is_monotone)


def thresholds_are_sane():
    for s in (0.0, 0.5, 1.0, -5.0, 99.0):
        t = detect.thresholds(s)
        assert t["explosion_multiplier"] > 1.0, t
        assert t["stagnation_frac"] > 0, t
        assert -1.0 <= t["oscillation_correlation"] <= 0.0, t
        assert t["oscillation_flip_threshold"] >= 1, t


check("thresholds stay sane, even off the end of the dial", thresholds_are_sane)


# ------------------------------------------------------------------ cost
def fast_on_long_runs():
    eng = engine()
    values = [2.0 * math.exp(-3.0 * i / 50000) for i in range(50000)]
    start = time.time()
    for i in range(49900, 50001, 10):
        eng.update({"loss": values[:i], "acc": values[:i]}, step=i)
    elapsed = time.time() - start
    assert elapsed < 5.0, "50k-reading history took %.1fs for 11 updates" % elapsed


check("a 50k-step history stays fast", fast_on_long_runs)


def no_mutation():
    """The engine must not edit the caller's training history."""
    original = {"loss": [1.0, float("nan"), 0.8] + [0.5] * 30, "acc": [0.1] * 33}
    copy = {k: list(v) for k, v in original.items()}
    engine().update(original, step=33)
    assert original == copy, "the engine modified the history it was given"


check("the caller's history is never modified", no_mutation)

print("\n".join("PASS  " + p for p in PASS))
if FAIL:
    print("\n".join("FAIL  " + f for f in FAIL))
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
