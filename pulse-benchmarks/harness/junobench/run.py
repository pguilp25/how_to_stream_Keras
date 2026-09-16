"""JunoBench x Pulse harness (run on the PC from ~/junobench).

    .venv/bin/python harness/run.py baseline  # crashes reproduce, fixes run clean (no Pulse)
    .venv/bin/python harness/run.py pulse     # Pulse + agent on every case, in parallel
    .venv/bin/python harness/run.py verify    # re-run each Pulse-patched script without Pulse

Cases default to NBspecific_1..8; pass names after the command to pick others.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
NB2PY = os.path.join(ROOT, "harness", "nb2py.py")
BENCH = os.path.join(ROOT, "benchmark")
RUNS = os.path.abspath(os.environ.get("BENCH_RUNS", os.path.join(ROOT, "runs")))
BASELINE = os.path.join(ROOT, "baseline")
CONFIG = os.path.abspath(os.environ.get("BENCH_CONFIG", os.path.join(ROOT, "pulse_config.json")))

SCRIPT_TIMEOUT = 30 * 60
PULSE_TIMEOUT = int(os.environ.get("BENCH_PULSE_TIMEOUT", 60 * 60))

ENV = dict(
    os.environ,
    MPLBACKEND="Agg",
    BROWSER="/bin/true",
    PLOTLY_RENDERER="json",
    PYTHONUNBUFFERED="1",
    TF_CPP_MIN_LOG_LEVEL="2",
    PULSE_TELEMETRY="off",
    PULSE_CONFIG=CONFIG,
)
if os.environ.get("BENCH_TRACE") == "1":
    ENV["PYTHONPATH"] = os.path.join(ROOT, "harness", "trace")
ENV.pop("DISPLAY", None)
ENV.pop("WAYLAND_DISPLAY", None)


def expected_errors():
    ws = openpyxl.load_workbook(os.path.join(ROOT, "benchmark_desc.xlsx"), read_only=True).active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    return {r[0]: dict(zip(header, r)) for r in rows if r[0]}


def convert(nb, out, *flags):
    subprocess.run([PY, NB2PY, nb, out, *flags], check=True)


def run(argv, cwd, log_path, timeout):
    """Run in its own process group so a timeout also kills anything it
    spawned (Pulse re-launches the script as a child after each fix)."""
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


def tail_error(log_path):
    """Last 'SomethingError: message' line of a log, if any."""
    with open(log_path, errors="ignore") as f:
        lines = [l.strip() for l in f if "Error" in l.split(":", 1)[0] and ":" in l]
    return lines[-1][:200] if lines else None


def baseline_case(case, expected):
    out = os.path.join(BASELINE, case)
    os.makedirs(out, exist_ok=True)
    cwd = os.path.join(BENCH, case)
    result = {"case": case, "expected": f"{expected['ename']}: {expected['evalue']}"}
    for kind, nb, flags in (
        ("reproduced", f"{case}_reproduced.ipynb", ()),
        ("notebook_view", f"{case}_reproduced.ipynb", ("--notebook-view",)),
        ("fixed", f"{case}_fixed.ipynb", ()),
    ):
        script = os.path.join(out, f"{kind}.py")
        convert(os.path.join(cwd, nb), script, *flags)
        log = os.path.join(out, f"{kind}.log")
        code, secs = run([PY, script], cwd, log, SCRIPT_TIMEOUT)
        result[kind] = {"exit": code, "seconds": secs, "last_error": tail_error(log)}
    return result


def pulse_case(case):
    out = os.path.join(RUNS, case)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    src = os.path.join(BENCH, case)
    # Only the script and its inputs -- never the fixed notebook or README.
    for name in os.listdir(src):
        if name == "data" or name.endswith((".h5", ".keras", ".pkl")):
            os.symlink(os.path.join(src, name), os.path.join(out, name))
    script = os.path.join(out, "notebook.py")
    convert(os.path.join(src, f"{case}_reproduced.ipynb"), script, "--pulse", "--notebook-view")
    shutil.copy(script, os.path.join(RUNS, f"{case}.original.py"))
    code, secs = run([PY, script], out, os.path.join(RUNS, f"{case}.pulse.log"), PULSE_TIMEOUT)
    return {"case": case, "exit": code, "seconds": secs}


def verify_case(case):
    out = os.path.join(RUNS, case)
    with open(os.path.join(out, "notebook.py")) as f:
        text = "".join(l for l in f if "# [pulse-hook]" not in l)
    script = os.path.join(out, "verify.py")
    with open(script, "w") as f:
        f.write(text)
    log = os.path.join(RUNS, f"{case}.verify.log")
    code, secs = run([PY, script], out, log, SCRIPT_TIMEOUT)
    diff = subprocess.run(["diff", "-u", os.path.join(RUNS, f"{case}.original.py"), os.path.join(out, "notebook.py")],
                          capture_output=True, text=True).stdout
    with open(os.path.join(RUNS, f"{case}.diff"), "w") as f:
        f.write(diff)
    return {"case": case, "exit": code, "seconds": secs, "last_error": tail_error(log), "changed": bool(diff)}


def main():
    cmd = sys.argv[1]
    cases = sys.argv[2:] or [f"NBspecific_{i}" for i in range(1, 9)]
    if cmd == "baseline":
        exp = expected_errors()
        jobs = lambda c: baseline_case(c, exp[c])
    elif cmd == "pulse":
        os.makedirs(RUNS, exist_ok=True)
        jobs = pulse_case
    elif cmd == "verify":
        jobs = verify_case
    else:
        raise SystemExit(__doc__)
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        results = list(pool.map(jobs, cases))
    path = os.path.join(BASELINE if cmd == "baseline" else RUNS, f"{cmd}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
