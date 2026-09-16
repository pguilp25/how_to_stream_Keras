"""Build Deep4ge bug-fixing instances (run on the PC with the TF 2.15 venv).

Deep4ge ships correct seed programs plus AST mutation operators, not buggy
programs. For each (seed, operator) candidate this:
  1. generates up to MAX_VARIANTS distinct mutants with the Deep4ge operator,
  2. trains the (unparsed) seed and each mutant REPEATS times,
  3. keeps mutants that train clearly and consistently worse ("killed"),
then selects one instance per candidate row (worst mutant) and writes:
  instances/<name>/train.py + CustomCallback.py      (what an agent sees)
  answers/<name>/correct.py + meta.json                (held out, for scoring)

    ~/deep4ge_bench/.venv/bin/python make_instances.py
"""
import ast
import csv
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

BENCH = os.path.expanduser("~/deep4ge_bench")
REPO = os.path.join(BENCH, "deep4ge")
PY = os.path.join(BENCH, ".venv", "bin", "python")
CALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_callback.py")
SEEDS = os.path.join(REPO, "data", "seed_programs")
WORK = os.path.join(BENCH, "work", "screen")

sys.path.insert(0, REPO)
from src.operators import OPERATOR_CLASSES  # noqa: E402
from src.operators import activation_ops, weight_ops, regularization_ops, layer_ops  # noqa: E402


# Deep4ge's Weight/Activation/Regularization/Layer operators mutate a built
# Keras model object (apply_to_model), so they change nothing in a source
# file and can't produce a bug an agent could fix by editing code. These are
# the same operator definitions (same catalog, same parameter spaces) applied
# to the source instead. Codes are prefixed "SRC_" to keep that distinction.
class _SourceOp(ast.NodeTransformer):
    category = ""

    def mutate(self, source_code):
        tree = self.visit(ast.parse(source_code))
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    @staticmethod
    def _kwarg(call, name):
        for kw in call.keywords:
            if kw.arg == name:
                return kw
        return None


class SrcChangeWeightsInit(_SourceOp):
    """SRC_WCI (Weight): change/add a Dense kernel_initializer."""
    category = "Weight"

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "Dense" or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "Dense"):
            kw = self._kwarg(node, "kernel_initializer")
            new = random.choice(weight_ops.KERAS_INITIALIZERS)
            if kw is not None and isinstance(kw.value, ast.Constant) and kw.value.value != new:
                if random.random() < 0.5:
                    kw.value = ast.Constant(value=new)
            elif kw is None and random.random() < 0.5:
                node.keywords.append(ast.keyword(arg="kernel_initializer", value=ast.Constant(value=new)))
        return node


class SrcChangeActivation(_SourceOp):
    """SRC_ACH (Activation): swap an activation= string for a different one."""
    category = "Activation"

    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg == "activation" and isinstance(node.value, ast.Constant):
            candidates = [a for a in activation_ops.ACTIVATION_FUNCTIONS if a != node.value.value]
            if random.random() < 0.5:
                node.value = ast.Constant(value=random.choice(candidates))
        return node


class SrcRemoveActivation(_SourceOp):
    """SRC_ARM (Activation): replace a non-linear activation with linear."""
    category = "Activation"

    def visit_keyword(self, node):
        self.generic_visit(node)
        if (node.arg == "activation" and isinstance(node.value, ast.Constant)
                and node.value.value not in ("linear", "softmax") and random.random() < 0.5):
            node.value = ast.Constant(value="linear")
        return node


class SrcChangeDropout(_SourceOp):
    """SRC_RCD (Regularization): change a Dropout rate (0.125/0.25/0.75/1.0)."""
    category = "Regularization"

    def visit_Call(self, node):
        self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name == "Dropout" and node.args and isinstance(node.args[0], ast.Constant):
            new = random.choice([0.125, 0.25, 0.75, 1.0])
            if new != node.args[0].value:
                node.args[0] = ast.Constant(value=new)
        return node


class SrcChangeLearningRate(_SourceOp):
    """SRC_HLR (Hyperparameter): set an optimizer's learning rate to another
    value from Deep4ge's HLR range (1e-5 .. 1.0).

    Deep4ge's own HLR operator can't be used: it replaces the whole optimizer
    argument with `__import__('tensorflow.keras.backend').variable(<lr>)`,
    which is not a valid optimizer, so the program dies at compile() instead
    of training worse.
    """
    category = "Hyperparameter"

    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg in ("learning_rate", "lr") and isinstance(node.value, ast.Constant):
            low, high = math.log10(1e-5), math.log10(1.0)
            node.value = ast.Constant(value=float(f"{10 ** random.uniform(low, high):.3g}"))
        return node


class SrcChangeBatchSize(_SourceOp):
    """SRC_HBS (Hyperparameter): change the training batch size."""
    category = "Hyperparameter"
    SIZES = [1, 2, 4, 8, 16, 32, 256, 512]

    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg == "batch_size" and isinstance(node.value, ast.Constant):
            node.value = ast.Constant(value=random.choice(self.SIZES))
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (len(node.targets) == 1 and getattr(node.targets[0], "id", "") == "batch_size"
                and isinstance(node.value, ast.Constant)):
            node.value = ast.Constant(value=random.choice(self.SIZES))
        return node


class SrcChangeNeurons(_SourceOp):
    """SRC_LCN (Layer): change a hidden Dense layer's unit count."""
    category = "Layer"

    def visit_Call(self, node):
        self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name == "Dense" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
            if node.args[0].value > 2 and random.random() < 0.5:  # never the 1-2 unit output layer
                node.args[0] = ast.Constant(value=random.choice(layer_ops.NEURON_COUNTS))
        return node


SOURCE_OPS = {
    "SRC_WCI": SrcChangeWeightsInit, "SRC_ACH": SrcChangeActivation,
    "SRC_ARM": SrcRemoveActivation, "SRC_RCD": SrcChangeDropout,
    "SRC_LCN": SrcChangeNeurons, "SRC_HLR": SrcChangeLearningRate,
    "SRC_HBS": SrcChangeBatchSize,
}


def operator(code):
    return SOURCE_OPS[code]() if code in SOURCE_OPS else OPERATOR_CLASSES[code]()


def category(code):
    return SOURCE_OPS[code].category if code in SOURCE_OPS else OPERATOR_CLASSES[code].category

REPEATS = 3
MAX_VARIANTS = 3
WORKERS = int(os.environ.get("WORKERS", 4))
TIMEOUT = 1200

# (instance name, seed file, operator, metric) -- metric "acc": mean of final
# train/val accuracy (higher is better); "loss": final val_loss (lower is
# better). Loss-function mutations (FLC) only use "acc" seeds, since they
# change the scale of the loss itself.
CANDIDATES = [
    ("sumsign_lr", "fnn/FNN_64151679_correct.py", "SRC_HLR", "loss"),
]


def model_name(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "main":
            if node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value)
    raise ValueError("no main('<name>') call")


def variants(source, code):
    base = ast.unparse(ast.parse(source))
    seen, out = {base}, []
    for k in range(40):
        random.seed(k)
        mutated = operator(code).mutate(source)
        if mutated not in seen:
            seen.add(mutated)
            out.append(mutated)
        if len(out) == MAX_VARIANTS:
            break
    return base, out


def train(source, tag):
    """Run a program once in a scratch dir; return its final-epoch row or None."""
    run_dir = tempfile.mkdtemp(prefix=tag + "_", dir=WORK)
    shutil.copy(CALLBACK, os.path.join(run_dir, "CustomCallback.py"))
    with open(os.path.join(run_dir, "train.py"), "w") as f:
        f.write(source)
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0")
    try:
        subprocess.run([PY, "train.py"], cwd=run_dir, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    path = os.path.join(run_dir, model_name(source) + ".csv")
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    shutil.rmtree(run_dir, ignore_errors=True)
    if not rows:
        return "collapsed"  # never produced an epoch at all
    return rows[-1]


def score(row, metric):
    """Final-epoch metric; "collapsed" (training died / never logged) scores
    as the worst possible value rather than being discarded -- a fault that
    kills training is the strongest fault, not a missing data point."""
    if row == "collapsed":
        return 0.0 if metric == "acc" else math.inf

    def num(key):
        try:
            v = float(row[key])
        except (TypeError, ValueError, KeyError):
            return math.nan
        return v
    if metric == "acc":
        v = (num("train_acc") + num("val_acc")) / 2
        return v if not math.isnan(v) else 0.0
    v = num("val_loss")
    return v if math.isfinite(v) else math.inf


def killed(base, mutant, metric):
    """Consistently and clearly worse than the correct program."""
    if any(math.isinf(m) for m in mutant) and not any(math.isinf(b) for b in base):
        return True  # training collapsed under the fault but not without it
    if metric == "acc":
        return statistics.mean(mutant) <= statistics.mean(base) - 0.10 and max(mutant) < min(base)
    return statistics.mean(mutant) >= 2 * statistics.mean(base) and min(mutant) > max(base)


def main():
    os.makedirs(WORK, exist_ok=True)
    jobs = {}  # (tag, idx) -> source
    plan = []
    for name, seed, code, metric in CANDIDATES:
        with open(os.path.join(SEEDS, seed)) as f:
            source = f.read()
        base, muts = variants(source, code)
        plan.append((name, seed, code, metric, base, muts))
        jobs[("base:" + seed, 0)] = base
        for i, m in enumerate(muts):
            jobs[(f"{name}:{i}", i)] = m
    tasks = [(key, src, r) for key, src in jobs.items() for r in range(REPEATS)]
    print(f"{len(tasks)} training runs", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(lambda t: (t[0], train(t[1], t[0][0].split(":")[0].replace("/", "_"))), tasks))
    results = {}
    for key, row in rows:
        results.setdefault(key, []).append(row)

    selected = []
    report = []
    for name, seed, code, metric, base, muts in plan:
        base_scores = [score(r, metric) for r in results[("base:" + seed, 0)] if r]
        best = None
        for i, m in enumerate(muts):
            runs = results[(f"{name}:{i}", i)]
            if len(base_scores) < REPEATS or any(r is None for r in runs):
                report.append({"name": name, "variant": i, "status": "incomplete"})
                continue
            if any(s == math.inf for s in base_scores) or (metric == "acc" and all(s == 0.0 for s in base_scores)):
                report.append({"name": name, "variant": i, "status": "seed does not train"})
                continue
            mscores = [score(r, metric) for r in runs]
            ok = killed(base_scores, mscores, metric)
            report.append({"name": name, "variant": i, "killed": ok, "base": base_scores, "mutant": mscores})
            gap = (statistics.mean(base_scores) - statistics.mean(mscores)) if metric == "acc" else (
                statistics.mean(mscores) / max(statistics.mean(base_scores), 1e-12))
            if ok and (best is None or gap > best[0]):
                best = (gap, i, mscores)
        if best:
            selected.append((name, seed, code, metric, base, muts[best[1]], base_scores, best[2]))

    inst_root, ans_root = os.path.join(BENCH, "instances"), os.path.join(BENCH, "answers")
    for name, seed, code, metric, base, mutant, base_scores, mscores in selected:
        inst, ans = os.path.join(inst_root, name), os.path.join(ans_root, name)
        for d in (inst, ans):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        shutil.copy(CALLBACK, os.path.join(inst, "CustomCallback.py"))
        with open(os.path.join(inst, "train.py"), "w") as f:
            f.write(mutant + "\n")
        with open(os.path.join(ans, "correct.py"), "w") as f:
            f.write(base + "\n")
        with open(os.path.join(ans, "meta.json"), "w") as f:
            json.dump({"seed": seed, "operator": code, "category": category(code),
                       "metric": metric, "base_scores": base_scores, "mutant_scores": mscores}, f, indent=2)
    with open(os.path.join(BENCH, "screen_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=1, default=str))
    print("SELECTED:", [s[0] for s in selected])


if __name__ == "__main__":
    main()
