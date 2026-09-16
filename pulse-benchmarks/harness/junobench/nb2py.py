"""Linearize a JunoBench notebook into a plain script, following the same
rules as JunoBench's auto_notebook_executer.py: only cells with an
execution_count run, sorted by it, and a cell whose first line carries
[re-execute]/[reexecute] runs twice in a row.

    python nb2py.py NB.ipynb OUT.py [--pulse] [--notebook-view]

--pulse          prepend the Pulse hook (auto_track) so the script runs under Pulse.
--notebook-view  also keep the never-executed cells that sit before the last
                 executed one, commented out, in notebook order -- what a
                 person looking at the notebook sees. Runs exactly the same
                 code. Only valid when execution order matches cell order.
"""
import json
import sys

PULSE_HEADER = "from pulse import auto_track  # [pulse-hook]\nauto_track()  # [pulse-hook]\n"


def is_reexecute(src):
    first = src.strip().splitlines()[0] if src.strip() else ""
    return "[re-execute]" in first or "[reexecute]" in first


def sanitize(src):
    # IPython-only syntax (magics / shell escapes) is not valid Python.
    # JunoBench's own [re-execute] annotation lines are dropped: they are
    # benchmark bookkeeping that states the root cause outright
    # ("this crash is because this cell was executed more than once").
    lines = []
    for line in src.splitlines():
        if "[re-execute]" in line or "[reexecute]" in line:
            continue
        if line.lstrip().startswith(("%", "!")):
            lines.append(line.replace(line.lstrip(), "# [ipython] " + line.lstrip(), 1))
        else:
            lines.append(line)
    return "\n".join(lines)


def blocks(nb, notebook_view):
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed = [c for c in code if c.get("execution_count") is not None]
    counts = [c["execution_count"] for c in executed]
    if notebook_view and counts != sorted(counts):
        raise SystemExit("--notebook-view needs execution order == cell order")
    if notebook_view:
        last = max(i for i, c in enumerate(code) if c.get("execution_count") is not None)
        cells = code[: last + 1]
    else:
        cells = sorted(executed, key=lambda c: c["execution_count"])
    for c in cells:
        src = "".join(c["source"])
        if c.get("execution_count") is None:
            if src.strip():
                body = "\n".join("# " + line for line in sanitize(src).splitlines())
                yield f"# %% [cell not executed]\n{body}"
            continue
        yield f"# %%\n{sanitize(src)}"
        if is_reexecute(src):
            yield f"# %%\n{sanitize(src)}"


def main():
    src_path, out_path = sys.argv[1], sys.argv[2]
    with open(src_path, encoding="utf-8") as f:
        nb = json.load(f)
    parts = [PULSE_HEADER if "--pulse" in sys.argv else "", "from IPython.display import display\n"]
    parts += ["\n" + b + "\n" for b in blocks(nb, "--notebook-view" in sys.argv)]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
