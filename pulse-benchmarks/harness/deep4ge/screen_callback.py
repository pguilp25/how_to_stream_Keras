"""Metrics-only stand-in for Deep4ge's EnhancedLoggingCallback.

Same import name, same constructor, same per-epoch CSV, but it only records
what training reports. Deep4ge's own callback recomputes gradients over three
extra batches for every training batch, which makes each run several times
slower without changing anything about training itself (it never applies what
it computes). Faults, programs and metrics are unaffected.
"""
import csv

import numpy as np
from keras.callbacks import Callback

HEADERS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]


class EnhancedLoggingCallback(Callback):
    def __init__(self, train_dataset, filename, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.filename = filename
        with open(self.filename, "w", newline="") as f:
            csv.writer(f).writerow(HEADERS)

    @staticmethod
    def _accuracy(logs, prefix=""):
        for key in ("accuracy", "acc", "categorical_accuracy", "sparse_categorical_accuracy"):
            if prefix + key in logs:
                return logs[prefix + key]
        mape = logs.get(prefix + "mean_absolute_percentage_error")
        return 1 - mape / 100 if mape is not None else np.nan

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = [epoch, logs.get("loss", np.nan), self._accuracy(logs),
               logs.get("val_loss", np.nan), self._accuracy(logs, "val_")]
        with open(self.filename, "a", newline="") as f:
            csv.writer(f).writerow(row)
