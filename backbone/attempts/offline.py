#!/usr/bin/env python3
"""Offline test: can what the search knows at the time of an attempt predict whether the attempt pays off?

    offline.py DATA.parquet [--label useful_wip] [--split seed|level]

Compared on held-out runs (never the same run in train and test):
  base   the label's rate
  kind   the objective kind's rate in the training runs (what a per-objective weight table can know)
  cheap  boosted trees on objective + ETA + running counts (no state)
  state  boosted trees with the start node's state added
Reported: AUC, average precision, and the share of paying attempts found in the best-ranked 20% of compute
frames (an attempt's cost is the emulator frames its GA used). Logged to MLflow (experiment sm64-attempt-value)."""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('--label', default='useful_wip')
ap.add_argument('--split', default='seed'); ap.add_argument('--cfg', default=''); A = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri('sqlite:///' + os.path.join(HERE, '..', 'mlflow.db')); mlflow.set_experiment('sm64-attempt-value')
df = pd.read_parquet(A.data)
if A.cfg: df = df[df.cfg.str.match(A.cfg)]
df = df[df.frames > 0].reset_index(drop=True)
df['kind_rate'] = (df.kind_s + 1) / (df.kind_n + 2)
df['log_frames_before'] = np.log10(df.frames_before + 1)
y = df[A.label].to_numpy().astype(int)
if A.split == 'seed':      # hold out one seed (or one level of a per-level batch) per configuration
    last = df.groupby('cfg').seed.transform('max'); test = (df.seed == last).to_numpy() & (df.groupby('cfg').seed.transform('nunique') > 1).to_numpy()
else:                      # hold out whole levels: every attempt whose start node is in a held-out level
    lv = sorted(df.level.unique()); held = set(lv[::4]); test = df.level.isin(held).to_numpy()
tr, te = ~test, test
# Categories as numbers: each value's rate of the label in the training runs (unseen values get the mean).
for c in ['kind', 'tech', 'action', 'via']:
    r = pd.Series(y[tr]).groupby(df[c][tr].to_numpy()).mean()
    df[c + '_r'] = df[c].map(r).astype(float).fillna(y[tr].mean())
if tr.sum() > 120000:      # a sample of the training attempts is enough for trees of this size
    keep = np.random.default_rng(1).random(len(df)) < 120000 / tr.sum(); tr = tr & keep
CHEAP = ['kind_r', 'tech_r', 'start_eta', 'stage', 'chain', 'plan_len', 'is_b', 'is_start', 'kind_n', 'kind_rate', 'kn_n', 'node_n', 'log_frames_before']
STATE = CHEAP + ['level', 'area', 'depth', 'x', 'y', 'z', 'stars', 'keys', 'coins', 'red', 'route_frame', 'hspeed', 'fwd_vel', 'action_r', 'via_r']


def top_share(score, y, cost, frac=0.2):
    """Share of paying attempts among those ranked best by score per frame, up to `frac` of all frames."""
    o = np.argsort(-score / cost); c = np.cumsum(cost[o]); k = np.searchsorted(c, frac * c[-1])
    return y[o][:k].sum() / max(1, y.sum())


def report(name, s):
    cost = df.frames.to_numpy()[te].astype(float)
    m = dict(auc=roc_auc_score(y[te], s), ap=average_precision_score(y[te], s), top20=top_share(s, y[te], cost))
    print(f'{name:6} AUC {m["auc"]:.3f}  AP {m["ap"]:.3f}  paying attempts in best 20% of frames {m["top20"]:.3f}')
    return m


with mlflow.start_run(run_name=f'offline-{A.label}-{A.split}' + (f'-{A.cfg}' if A.cfg else '')):
    mlflow.log_params(dict(label=A.label, split=A.split, cfg=A.cfg, n_train=int(tr.sum()), n_test=int(te.sum()), runs=int(df.run.nunique()), data=os.path.basename(A.data)))
    cost_all = df.frames.to_numpy().astype(float)
    print(f'{A.label}, split by {A.split}: train {tr.sum()} test {te.sum()} attempts; rate {y[te].mean():.4f}; '
          f'share of compute frames on paying attempts {cost_all[te][y[te] == 1].sum() / cost_all[te].sum():.3f}')
    mlflow.log_metric('rate', y[te].mean()); mlflow.log_metric('frames_share_paying', cost_all[te][y[te] == 1].sum() / cost_all[te].sum())
    res = {}
    res['base'] = report('base', np.random.default_rng(0).random(te.sum()))
    res['kind'] = report('kind', df.kind_r[te].to_numpy())
    res['eta'] = report('eta', -df.start_eta[te].fillna(1e6).to_numpy())
    for name, cols in [('cheap', CHEAP), ('state', STATE)]:
        m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        m.fit(df.loc[tr, cols], y[tr])
        res[name] = report(name, m.predict_proba(df.loc[te, cols])[:, 1])
    for n, m in res.items():
        for k, v in m.items(): mlflow.log_metric(f'{n}_{k}', float(v))
