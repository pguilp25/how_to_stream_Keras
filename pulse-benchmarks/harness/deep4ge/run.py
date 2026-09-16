"""Deep4ge x Pulse / Claude Code harness (run on the PC from ~/deep4ge_bench).

    .venv/bin/python run.py pulse [names...]   # Pulse + agent on each instance
    .venv/bin/python run.py score [names...]   # retrain patched scripts, compare to the correct seed

Instances come from instances/<name>/ (built by make_instances.py); the correct
program and thresholds live in answers/<name>/ and are never copied into a run
directory.
"""
import ast
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(BENCH) != "deep4ge_bench":  # harness lives outside the bench dir
    BENCH = os.path.expanduser("~/deep4ge_bench")
PY = os.path.join(BENCH, ".venv", "bin", "python")
HARNESS = os.path.dirname(os.path.abspath(__file__))
INSTANCES = os.path.join(BENCH, "instances")
ANSWERS = os.path.join(BENCH, "answers")
RUNS = os.path.abspath(os.environ.get("BENCH_RUNS", os.path.join(BENCH, "runs")))
CONFIG = os.path.abspath(os.environ.get("BENCH_CONFIG", os.path.expanduser("~/junobench/pulse_config.json")))

QUESTION = (
    "This Keras training script has one injected bug that makes the model train worse than it "
    "should. Find the root cause and fix train.py with the smallest correct change."
)
PULSE_TIMEOUT = int(os.environ.get("BENCH_PULSE_TIMEOUT", 3600))
SCORE_TIMEOUT = 1800
SCORE_REPEATS = 3
HOOK = "# [pulse-hook]"

ENV = dict(
    os.environ, TF_CPP_MIN_LOG_LEVEL="2", CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg",
    PYTHONUNBUFFERED="1", PULSE_TELEMETRY="off", PULSE_CONFIG=CONFIG,
)
if os.environ.get("BENCH_TRACE") == "1":
    ENV["PYTHONPATH"] = os.path.join(os.path.expanduser("~/junobench"), "harness", "trace")


def names(argv):
    return argv or sorted(os.listdir(INSTANCES))


def add_pulse_hook(source):
    """`from pulse import auto_track` + `auto_track()` as the first statement of
    the __main__ block, i.e. immediately before training starts."""
    tree = ast.parse(source)
    tree.body.insert(0, ast.ImportFrom(module="pulse", names=[ast.alias(name="auto_track")], level=0))
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and getattr(node.test.left, "id", "") == "__name__"):
            node.body.insert(0, ast.Expr(ast.Call(func=ast.Name(id="auto_track", ctx=ast.Load()), args=[], keywords=[])))
    ast.fix_missing_locations(tree)
    lines = ast.unparse(tree).splitlines()
    return "\n".join(l + f"  {HOOK}" if "auto_track" in l else l for l in lines) + "\n"


def run(argv, cwd, log_path, timeout):
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(argv, cwd=cwd, env=ENV, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = "timeout"
    return code, round(time.time() - start)


def pulse_case(name):
    out = os.path.join(RUNS, name)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(os.path.join(INSTANCES, name), out)
    with open(os.path.join(out, "train.py")) as f:
        original = f.read()
    with open(os.path.join(out, "train.py"), "w") as f:
        f.write(add_pulse_hook(original))
    shutil.copy(os.path.join(out, "train.py"), os.path.join(RUNS, f"{name}.original.py"))
    code, secs = run([PY, os.path.join(HARNESS, "driver.py"), "train.py", QUESTION],
                     out, os.path.join(RUNS, f"{name}.pulse.log"), PULSE_TIMEOUT)
    return {"case": name, "exit": code, "seconds": secs}


def train_once(name, source, tag, index):
    """Train a program in its own scratch dir; return the final-epoch metric."""
    work = os.path.join(RUNS, f"{name}.score", f"{tag}{index}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    shutil.copy(os.path.join(INSTANCES, name, "CustomCallback.py"), work)
    with open(os.path.join(work, "train.py"), "w") as f:
        f.write(source)
    run([PY, "train.py"], work, os.path.join(work, "train.log"), SCORE_TIMEOUT)
    sys.path.insert(0, HARNESS)
    from make_instances import model_name, score  # same parsing/metric as the builder
    import csv
    import glob
    meta = json.load(open(os.path.join(ANSWERS, name, "meta.json")))
    # Find whatever per-epoch log the run wrote: a patch may rename it (or
    # change the epoch count), which must not be read as "no result".
    expected = os.path.join(work, model_name(source) + ".csv")
    candidates = [expected] if os.path.exists(expected) else sorted(
        glob.glob(os.path.join(work, "*.csv")), key=os.path.getmtime, reverse=True)
    rows = []
    for path in candidates:
        rows = list(csv.DictReader(open(path)))
        if rows:
            break
    if not rows:
        return "collapsed"  # trained nothing at all -> worst possible score
    return score(rows[-1], meta["metric"])


def loss_functions(source):
    """Loss arguments passed to model.compile() -- if a patch changes these,
    a val_loss comparison against the correct program is meaningless (Pulse's
    start-of-run lint rewrites e.g. 'mse' + accuracy metric on its own)."""
    found = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "compile":
            for kw in node.keywords:
                if kw.arg == "loss":
                    found.append(ast.unparse(kw.value))
            for arg in node.args:
                found.append(ast.unparse(arg))
    return found


def score_case(name):
    meta = json.load(open(os.path.join(ANSWERS, name, "meta.json")))
    with open(os.path.join(RUNS, name, "train.py")) as f:
        patched = "".join(l for l in f if HOOK not in l)
    with open(os.path.join(RUNS, f"{name}.original.py")) as f:
        original = "".join(l for l in f if HOOK not in l)
    changed = patched != original
    correct = open(os.path.join(ANSWERS, name, "correct.py")).read()
    try:
        reverted = ast.dump(ast.parse(patched)) == ast.dump(ast.parse(correct))
    except SyntaxError:
        reverted = False
    raw = [train_once(name, patched, "patched", i) for i in range(SCORE_REPEATS)] if changed else []
    sys.path.insert(0, HARNESS)
    from make_instances import score as score_row
    scores = [score_row(r, meta["metric"]) if r == "collapsed" else r for r in raw if r is not None]
    base = meta["base_scores"]
    if not scores:
        restored = False
    elif meta["metric"] == "acc":
        restored = statistics.mean(scores) >= statistics.mean(base) - 0.05
    else:
        # Absolute floor as well as the ratio: a correct program whose loss is
        # ~0 would otherwise need a loss of <= 0 to count as restored.
        restored = statistics.mean(scores) <= max(1.5 * max(base), max(base) + 0.05)
    diff = subprocess.run(["diff", "-u", os.path.join(RUNS, f"{name}.original.py"),
                           os.path.join(RUNS, name, "train.py")], capture_output=True, text=True).stdout
    with open(os.path.join(RUNS, f"{name}.diff"), "w") as f:
        f.write(diff)
    loss_changed = loss_functions(patched) != loss_functions(correct)
    return {"case": name, "category": meta["category"], "operator": meta["operator"],
            "metric": meta["metric"], "changed": changed, "reverted_exactly": reverted,
            "restored": restored and not (meta["metric"] == "loss" and loss_changed),
            "loss_fn_changed": loss_changed, "patched_scores": scores,
            "base_scores": base, "buggy_scores": meta["mutant_scores"]}


def main():
    cmd = sys.argv[1]
    cases = names(sys.argv[2:])
    os.makedirs(RUNS, exist_ok=True)
    job = {"pulse": pulse_case, "score": score_case}[cmd]
    workers = int(os.environ.get("WORKERS", len(cases)))
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(job, cases))
    with open(os.path.join(RUNS, f"{cmd}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
