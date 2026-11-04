"""MLflow calls that never kill a run: the shared SQLite store locks when
several runs (and the UI) write at once, so each call retries, then gives up
with a message."""

import time

import mlflow


def safe(fn, *args, tries=8, **kw):
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001  (database is locked, ...)
            if i == tries - 1:
                print(f"mlflow: giving up on {fn.__name__}: {e}", flush=True)
                return None
            time.sleep(0.5 * 2**i)


def log_metrics(m, step):
    return safe(mlflow.log_metrics, m, step=step)


def log_metric(k, v, step):
    return safe(mlflow.log_metric, k, v, step=step)
