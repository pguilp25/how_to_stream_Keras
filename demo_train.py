"""Train a small MNIST convnet with LiveTelemetry attached; open http://127.0.0.1:8765/."""

import argparse
import os
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

from telemetry import LiveTelemetry

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--port", type=int, default=8765)
p.add_argument("--hold", action="store_true", help="keep serving after training ends")
args = p.parse_args()

(x, y), (xv, yv) = keras.datasets.mnist.load_data()
x = (x[..., None] / 255.0).astype("float32")
xv = (xv[..., None] / 255.0).astype("float32")

model = keras.Sequential([
    keras.Input((28, 28, 1)),
    keras.layers.Conv2D(16, 3, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10),
], name="mnist_cnn")

steps = args.epochs * int(np.ceil(len(x) / args.batch_size))
model.compile(
    optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(2e-3, steps), clipnorm=1.0),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

tel = LiveTelemetry(port=args.port, stats_every=10, hist_every=50, probe_x=xv[:128])
model.fit(x, y, validation_data=(xv, yv), epochs=args.epochs, batch_size=args.batch_size,
          callbacks=[tel], verbose=2)

if args.hold:
    print("[telemetry] training done; still serving, Ctrl+C to exit")
    while True:
        time.sleep(3600)
