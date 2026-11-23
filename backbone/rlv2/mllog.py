"""MLflow for the rlv2 trainers: experiment sm64-rl, through sm64bb.mlog (which
retries on SQLite locks and never kills a run). `start` and `log` never raise.
Backfill finished or running logs: python rlv2/mllog.py <run_name> <log> [...]."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BB = os.path.dirname(HERE)
sys.path.insert(0, BB)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def start(name, params):
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BB, 'mlflow.db')}")
        if mlflow.get_experiment_by_name("sm64-rl") is None:
            mlflow.create_experiment("sm64-rl", artifact_location=os.path.join(BB, "mlartifacts"))
        mlflow.set_experiment("sm64-rl")
        mlflow.start_run(run_name=name)
        mlflow.log_params({k: str(v)[:400] for k, v in params.items() if k != "rt"})
    except Exception as e:
        print("mlflow start failed:", e, flush=True)


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        k2 = (pre + "/" + str(k)).lstrip("/").replace("@", "_")
        if isinstance(v, dict):
            out.update(flat(v, k2))
        elif isinstance(v, bool):
            out[k2] = float(v)
        elif isinstance(v, (int, float)):
            out[k2] = float(v)
    return out


def log(rec, step, pre=""):
    try:
        from sm64bb import mlog

        mlog.log_metrics(flat(rec, pre), step=int(step))
    except Exception as e:
        print("mlflow log failed:", e, flush=True)


def stop():
    try:
        import mlflow

        mlflow.end_run()
    except Exception:
        pass


if __name__ == "__main__":
    # Backfill: one run per log; JSON lines with "frames" (RL) or "step" (BC).
    name = sys.argv[1]
    start(name + "-backfill", {"backfilled_from": sys.argv[2]})
    for line in open(sys.argv[2]):
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        step = r.get("frames", r.get("step"))
        if step is not None:
            log(r, step)
    stop()
