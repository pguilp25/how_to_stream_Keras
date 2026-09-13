# how_to_stream_Keras

Stream everything about a Keras 3 training run into an app, live, while `fit()` runs:
loss and metrics per step, learning rate, per-variable weight / gradient / update stats,
histograms, layer activations, and on-demand full tensors.

One callback, standard library HTTP server, no extra dependencies beyond Keras + NumPy.

## Install

```bash
python3 -m venv ~/keras-env
~/keras-env/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
~/keras-env/bin/pip install keras numpy
```

Use the torch backend (`KERAS_BACKEND=torch`) if you want gradients; see [Gradients](#gradients).

## Use

```python
from telemetry import LiveTelemetry

tel = LiveTelemetry(port=8765, stats_every=10, hist_every=50, probe_x=x_val[:128])
model.fit(x, y, validation_data=(x_val, y_val), callbacks=[tel])
```

Demo (MNIST convnet + live dashboard at http://127.0.0.1:8765/):

```bash
KERAS_BACKEND=torch ~/keras-env/bin/python demo_train.py --epochs 3 --hold
```

## Consuming it from an app

| Endpoint | Returns |
|---|---|
| `GET /api/stream` | Server-Sent Events, one JSON event per message |
| `GET /api/state` | run info + every event so far (for apps that connect mid-run) |
| `GET /api/tensor?path=P` | full current value of variable `P`; add `&grad=1` for its gradient. Captured between steps, never torn |
| `GET /` | the demo dashboard |

In-process instead of (or as well as) HTTP: `LiveTelemetry(port=None, sink=my_function)` calls
`my_function(event_dict)` for every event.

To join mid-run without gaps (what `dashboard.html` does): open the stream first and buffer,
fetch `/api/state`, replay it, then replay the buffer skipping events with `t` <= the last seen.

```js
const es = new EventSource("http://127.0.0.1:8765/api/stream");
es.onmessage = m => { const ev = JSON.parse(m.data); if (ev.type === "batch") console.log(ev.step, ev.logs_step.loss); };
```

## Events

Every event has `type` and `t` (unix time). NaN/inf are sent as `null`.

| type | when | fields |
|---|---|---|
| `run_begin` | `fit()` starts | backend, versions, fit params (epochs, steps), compile config (optimizer, loss, metrics), model JSON, per-layer table, every variable (path, shape, dtype, trainable), param counts |
| `epoch_begin` | each epoch | epoch |
| `batch` | every step | epoch, batch, step, lr, `logs`, `logs_step`, dt, steps_per_sec |
| `stats` | every `stats_every` steps | per variable: weight mean/std/min/max/l2, grad l2/mean_abs/max_abs, update l2, update/weight ratio; global grad norm; non-trainable float vars (e.g. BatchNorm moving stats) |
| `hist` | every `hist_every` steps | weight and gradient histograms per trainable variable |
| `activations` | every `hist_every` steps, if `probe_x` given | per-layer output mean/std/min/max/fraction zero + histogram |
| `epoch_end` | each epoch | epoch, logs (incl. `val_*`), seconds |
| `run_end` | `fit()` ends | final logs, total seconds |

## Things to know

- **`logs` are running means.** Keras reports batch loss/metrics averaged since the start of
  the epoch (metrics reset each epoch). `logs_step` un-averages them into that step's own
  values (exact for equal batch sizes; verified to ~1e-6 against per-batch ground truth).
- <a id="gradients"></a>**Gradients** are only available on the torch backend. Keras gives
  callbacks no gradient hook, but torch's trainer calls `zero_grad()` at the *start* of each
  step, so after a step `variable.value.grad` still holds that step's gradient (raw: before
  clipping, loss-scaled if using `LossScaleOptimizer`). On TensorFlow/JAX the gradients never
  leave `train_step`, so those fields are omitted (those backends are untested).
- **Overhead** measured on CPU with a small convnet: ~2 ms/step average on a ~43 ms step.
  The activation probe is the expensive part (~70 ms each time); raise `hist_every` for large
  models or drop `probe_x`.
- Activations need a Functional/Sequential model (subclassed models have no symbolic graph;
  the callback prints a notice and skips them).
- The server binds `127.0.0.1` by default; pass `host="0.0.0.0"` to reach it from another machine.
