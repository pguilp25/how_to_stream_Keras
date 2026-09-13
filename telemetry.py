"""Live training telemetry for Keras 3: stream everything about a run into an app.

    from telemetry import LiveTelemetry
    tel = LiveTelemetry(port=8765, stats_every=10, hist_every=50, probe_x=x[:64])
    model.fit(x, y, callbacks=[tel])

While fit() runs, any app can consume:
    GET  /api/stream            Server-Sent Events, one JSON event per message
    GET  /api/state             everything so far (run info + event backlog) for late joiners
    GET  /api/tensor?path=P     full current value of one variable (or its gradient: &grad=1),
                                snapshotted between steps so it is never torn
    GET  /                      demo dashboard (dashboard.html)
Or pass sink=callable to receive the event dicts in-process instead of / as well as HTTP.

Event types (all JSON, every event has "type" and "t" = unix time):
  run_begin   backend, versions, fit params (epochs, steps), compile config (optimizer, loss,
              metrics), model config + JSON, per-layer table (class, output shape, params),
              every variable (path, shape, dtype, trainable), param counts
  epoch_begin epoch
  batch       epoch, batch, step (optimizer.iterations), lr, logs (loss + metrics as Keras
              reports them: running means since epoch start), logs_step (the same metrics
              for this step alone), dt (step seconds), steps_per_sec
  stats       every `stats_every` steps, per variable: weight mean/std/min/max/l2, gradient
              l2/mean_abs/max_abs (torch backend), update l2 and update/weight ratio for that
              step, plus global grad norm; non-trainable vars (e.g. BatchNorm moving stats) too
  hist        every `hist_every` steps: weight + gradient histograms per trainable variable
  activations every `hist_every` steps if probe_x is given: per-layer output stats + histogram
  epoch_end   epoch, logs (incl. val_*), epoch seconds
  run_end     final logs, total seconds
NaN/inf become null so the stream is valid JSON.

Gradients: Keras exposes no gradient hook to callbacks. On the torch backend the trainer calls
zero_grad() at the *start* of each step, so right after a step `variable.value.grad` still holds
that step's raw (pre-clipping, loss-scaled if using LossScaleOptimizer) gradient. On TF/JAX the
gradients never leave train_step; those fields are simply omitted.
"""

import contextlib
import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import keras
from keras import ops


def _clean(obj):
    """Make obj JSON-safe: numpy -> python, NaN/inf -> None, unknown -> str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer, int, bool, str)) or obj is None:
        return obj.item() if isinstance(obj, np.generic) else obj
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return str(obj)


def _histogram(a, bins):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    counts, edges = np.histogram(a, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def _no_grad():
    """Keep telemetry math out of the torch autograd graph."""
    if keras.backend.backend() == "torch":
        import torch

        return torch.no_grad()
    return contextlib.nullcontext()


def _grad_of(variable):
    """The gradient left on a variable by the last train step (torch backend only)."""
    if keras.backend.backend() != "torch":
        return None
    return getattr(variable.value, "grad", None)


class _Hub:
    """Fan-out of events to SSE subscribers, plus the backlog for /api/state."""

    def __init__(self, max_backlog):
        self.lock = threading.Lock()
        self.subscribers = []
        self.run_info = None
        self.backlog = []
        self.max_backlog = max_backlog

    def publish(self, event):
        line = json.dumps(event)
        with self.lock:
            if event["type"] == "run_begin":
                self.run_info, self.backlog = event, []
            elif event["type"] != "hist" and event["type"] != "activations":
                self.backlog.append(event)
                if len(self.backlog) > self.max_backlog:
                    del self.backlog[: len(self.backlog) - self.max_backlog]
            for q in self.subscribers:
                try:
                    q.put_nowait(line)
                except queue.Full:  # slow client: drop rather than stall training
                    pass

    def subscribe(self):
        q = queue.Queue(maxsize=10000)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.subscribers.remove(q)


class LiveTelemetry(keras.callbacks.Callback):
    def __init__(
        self,
        port=8765,
        host="127.0.0.1",
        sink=None,
        stats_every=10,
        hist_every=100,
        hist_bins=40,
        probe_x=None,
        max_backlog=50000,
    ):
        super().__init__()
        self.sink = sink
        self.stats_every = stats_every
        self.hist_every = hist_every
        self.hist_bins = hist_bins
        self.probe_x = probe_x
        self.hub = _Hub(max_backlog)
        self._tensor_requests = queue.Queue()
        self._probe_model = None
        self._prev_weights = None
        self.server = None
        if port is not None:
            self.server = ThreadingHTTPServer((host, port), self._handler())
            self.server.daemon_threads = True
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            print(f"[telemetry] live at http://{host}:{port}/")

    # ---- emitting -------------------------------------------------------------------------

    def emit(self, type_, **fields):
        event = _clean({"type": type_, "t": time.time(), **fields})
        self.hub.publish(event)
        if self.sink is not None:
            self.sink(event)

    # ---- keras hooks ----------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        m = self.model
        self._t_train = time.time()
        layers = []
        for layer in m.layers:
            try:
                out_shape = layer.output.shape
            except (AttributeError, ValueError):
                out_shape = None
            layers.append({
                "name": layer.name,
                "class": type(layer).__name__,
                "output_shape": out_shape,
                "params": layer.count_params() if layer.built else None,
                "trainable": layer.trainable,
            })
        try:
            model_json = json.loads(m.to_json())
        except Exception:
            model_json = None
        self.emit(
            "run_begin",
            backend=keras.backend.backend(),
            keras_version=keras.__version__,
            params=self.params,
            compile_config=m.get_compile_config(),
            model_name=m.name,
            model_json=model_json,
            layers=layers,
            variables=[
                {"path": v.path, "shape": v.shape, "dtype": v.dtype, "trainable": v.trainable}
                for v in m.variables
            ],
            total_params=m.count_params(),
            trainable_params=sum(int(np.prod(v.shape)) for v in m.trainable_variables),
            gradients_available=keras.backend.backend() == "torch",
        )
        if self.probe_x is not None:
            try:
                inputs = m.inputs[0] if len(m.inputs) == 1 else m.inputs
                self._probe_model = keras.Model(inputs, [l.output for l in m.layers])
            except Exception as e:  # subclassed models have no symbolic graph
                print(f"[telemetry] activations disabled: {e}")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._t_epoch = time.time()
        self._running, self._n = {}, 0
        self.emit("epoch_begin", epoch=epoch)

    def on_train_batch_begin(self, batch, logs=None):
        self._t_batch = time.time()
        step = self._step() + 1  # the step about to run
        if self.stats_every and step % self.stats_every == 0:
            # snapshot so on_train_batch_end can report the exact update this step applied
            with _no_grad():
                self._prev_weights = [ops.copy(v.value) for v in self.model.trainable_variables]

    def on_train_batch_end(self, batch, logs=None):
        dt = time.time() - self._t_batch
        step = self._step()
        logs = logs or {}
        # Keras batch logs are running means since the epoch started (metrics reset each
        # epoch). Un-average them to get this step's own values (exact for equal batch sizes).
        n = batch + 1
        logs_step = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                prev = self._running.get(k)
                logs_step[k] = v if prev is None else (v * n - prev * self._n) / (n - self._n)
                self._running[k] = v
        self._n = n
        self.emit(
            "batch",
            epoch=self._epoch,
            batch=batch,
            step=step,
            lr=self._lr(),
            logs=logs,
            logs_step=logs_step,
            dt=dt,
            steps_per_sec=1 / dt if dt > 0 else None,
        )
        with _no_grad():
            if self._prev_weights is not None:
                self._emit_stats(step)
                self._prev_weights = None
            if self.hist_every and step % self.hist_every == 0:
                self._emit_hist(step)
                if self._probe_model is not None:
                    self._emit_activations(step)
            self._serve_tensor_requests()

    def on_epoch_end(self, epoch, logs=None):
        self.emit("epoch_end", epoch=epoch, logs=logs or {}, seconds=time.time() - self._t_epoch)

    def on_train_end(self, logs=None):
        self._serve_tensor_requests()
        self.emit("run_end", logs=logs or {}, seconds=time.time() - self._t_train)

    # ---- extraction -----------------------------------------------------------------------

    def _step(self):
        return int(ops.convert_to_numpy(self.model.optimizer.iterations))

    def _lr(self):
        try:  # evaluates LearningRateSchedules at the current iteration
            return float(ops.convert_to_numpy(self.model.optimizer.learning_rate))
        except Exception:
            return None

    @staticmethod
    def _l2(t):
        return float(ops.convert_to_numpy(ops.sqrt(ops.sum(ops.square(ops.cast(t, "float32"))))))

    def _emit_stats(self, step):
        per_var = {}
        grad_sq = 0.0
        trainable = self.model.trainable_variables
        for v, before in zip(trainable, self._prev_weights):
            w = ops.cast(v.value, "float32")
            w_l2 = self._l2(w)
            upd_l2 = self._l2(ops.subtract(w, ops.cast(before, "float32")))
            s = {
                "mean": float(ops.convert_to_numpy(ops.mean(w))),
                "std": float(ops.convert_to_numpy(ops.std(w))),
                "min": float(ops.convert_to_numpy(ops.min(w))),
                "max": float(ops.convert_to_numpy(ops.max(w))),
                "l2": w_l2,
                "update_l2": upd_l2,
                "update_ratio": upd_l2 / w_l2 if w_l2 > 0 else None,
            }
            g = _grad_of(v)
            if g is not None:
                g = ops.cast(g, "float32")
                s["grad_l2"] = self._l2(g)
                s["grad_mean_abs"] = float(ops.convert_to_numpy(ops.mean(ops.abs(g))))
                s["grad_max_abs"] = float(ops.convert_to_numpy(ops.max(ops.abs(g))))
                grad_sq += s["grad_l2"] ** 2
            per_var[v.path] = s
        for v in self.model.non_trainable_variables:
            if "float" not in str(v.dtype):  # skip RNG seed state etc.
                continue
            w = ops.cast(v.value, "float32")
            per_var[v.path] = {
                "mean": float(ops.convert_to_numpy(ops.mean(w))),
                "std": float(ops.convert_to_numpy(ops.std(w))),
                "l2": self._l2(w),
                "trainable": False,
            }
        self.emit(
            "stats",
            step=step,
            epoch=self._epoch,
            global_grad_l2=math.sqrt(grad_sq) if keras.backend.backend() == "torch" else None,
            variables=per_var,
        )

    def _emit_hist(self, step):
        out = {}
        for v in self.model.trainable_variables:
            h = {"weights": _histogram(ops.convert_to_numpy(v.value).ravel(), self.hist_bins)}
            g = _grad_of(v)
            if g is not None:
                h["grads"] = _histogram(ops.convert_to_numpy(g).ravel(), self.hist_bins)
            out[v.path] = h
        self.emit("hist", step=step, epoch=self._epoch, variables=out)

    def _emit_activations(self, step):
        outs = self._probe_model(self.probe_x, training=False)
        layers = {}
        for layer, o in zip(self.model.layers, outs):
            a = ops.convert_to_numpy(o).astype(np.float32).ravel()
            layers[layer.name] = {
                "mean": a.mean(), "std": a.std(), "min": a.min(), "max": a.max(),
                "frac_zero": float((a == 0).mean()),
                "hist": _histogram(a, self.hist_bins),
            }
        self.emit("activations", step=step, epoch=self._epoch, layers=layers)

    def _serve_tensor_requests(self):
        """Fulfil /api/tensor requests between steps, when no weight is mid-update."""
        by_path = {v.path: v for v in self.model.variables}
        while True:
            try:
                path, want_grad, reply = self._tensor_requests.get_nowait()
            except queue.Empty:
                return
            v = by_path.get(path)
            if v is None:
                reply.put({"error": f"no variable {path!r}", "paths": sorted(by_path)})
                continue
            t = _grad_of(v) if want_grad else v.value
            reply.put({
                "path": path, "step": self._step(), "grad": want_grad, "shape": v.shape,
                "values": None if t is None else ops.convert_to_numpy(t),
            })

    # ---- http -----------------------------------------------------------------------------

    def _handler(self):
        tel = self
        dashboard = Path(__file__).with_name("dashboard.html")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                url = urlparse(self.path)
                qs = parse_qs(url.query)
                if url.path == "/":
                    self._send(200, dashboard.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    with tel.hub.lock:
                        body = {"run": tel.hub.run_info, "events": list(tel.hub.backlog)}
                    self._send(200, body)
                elif url.path == "/api/tensor":
                    path = qs.get("path", [""])[0]
                    reply = queue.Queue()
                    tel._tensor_requests.put((path, qs.get("grad", ["0"])[0] == "1", reply))
                    try:  # answered at the next step boundary
                        self._send(200, _clean(reply.get(timeout=30)))
                    except queue.Empty:
                        self._send(504, {"error": "training not stepping (finished or paused)"})
                elif url.path == "/api/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    q = tel.hub.subscribe()
                    try:
                        while True:
                            try:
                                line = q.get(timeout=15)
                                self.wfile.write(f"data: {line}\n\n".encode())
                            except queue.Empty:
                                self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        tel.hub.unsubscribe(q)
                else:
                    self._send(404, {"error": "not found"})

        return Handler
