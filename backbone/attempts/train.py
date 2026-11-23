#!/usr/bin/env python3
"""Attempt-value model: one small network, several heads, trained on the attempt archive (build.py).

    train.py DATA.parquet OUT_DIR [--epochs 12] [--hidden 256] [--split seed|level] [--name NAME]

Input: what the search knows when it plans an attempt (the objective's kind and start ETA, the start state,
running counts of the run so far). Heads, on a shared trunk:
  success   the objective is met                               (BCE)
  pays      the subtree under the added node makes progress    (BCE; useful_wip)
  star      ... reaches a star or key                          (BCE; useful_star)
  cost      log10 emulator frames the GA uses, quantiles 0.1 / 0.5 / 0.9      (pinball)
  seg       log10 route frames of the segment when it succeeds, same quantiles (pinball, successes only)
  value     0.999 ^ (route frames from the start state to the first star or key under the added node), 0 when
            none: how soon a star follows this attempt, through what the search itself went on to do (BCE on
            the soft target)
Quantiles, not a mean and a variance: outcomes are two-peaked and heavy-tailed.
Logged to MLflow (experiment sm64-attempt-value). OUT_DIR gets model.pt and model.json (weights and the
feature recipe, for the engine's own forward pass)."""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('out')
ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--split', default='seed'); ap.add_argument('--name', default=''); ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--batch', type=int, default=4096); ap.add_argument('--no-state', action='store_true'); ap.add_argument('--all', action='store_true', help='train on every run (for the model the engine uses)')
A = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(A.out, exist_ok=True)
mlflow.set_tracking_uri('sqlite:///' + os.path.join(HERE, '..', 'mlflow.db')); mlflow.set_experiment('sm64-attempt-value')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_parquet(A.data); df = df[df.frames > 0].reset_index(drop=True)
if A.split == 'seed':
    last = df.groupby('cfg').seed.transform('max'); test = ((df.seed == last) & (df.groupby('cfg').seed.transform('nunique') > 1)).to_numpy()
else:
    lv = sorted(df.level.unique()); test = df.level.isin(set(lv[::4])).to_numpy()
train = np.ones(len(df), bool) if A.all else ~test

# Categories: an index per value seen at least 20 times in training, 0 otherwise.
CATS = ['kind', 'level'] + ([] if A.no_state else ['action'])
vocab = {}
for c in CATS:
    vc = df.loc[train, c].astype(str).value_counts(); vocab[c] = {v: i + 1 for i, v in enumerate(vc[vc >= 20].index)}
# Numbers: (name, transform). Each is standardised with the training mean and deviation.
NUM = [('start_eta', 'log1p'), ('stage', 'id'), ('plan_len', 'id'), ('kind_n', 'log1p'), ('kind_rate', 'id'), ('kn_n', 'log1p'), ('node_n', 'log1p'), ('frames_before', 'log1p'), ('eta_missing', 'id')]
if not A.no_state:
    NUM += [('y', 'id'), ('hspeed', 'id'), ('fwd_vel', 'id'), ('stars', 'id'), ('coins', 'id'), ('red', 'id'), ('route_frame', 'log1p'), ('depth', 'log1p')]
df['kind_rate'] = (df.kind_s + 1) / (df.kind_n + 2); df['eta_missing'] = df.start_eta.isna().astype(float); df['start_eta'] = df.start_eta.fillna(0).clip(0, 5000)
df['route_frame'] = df.route_frame.clip(lower=0)
X = np.stack([np.log1p(df[n].to_numpy(float)) if t == 'log1p' else df[n].to_numpy(float) for n, t in NUM], 1).astype(np.float32)
mu, sd = X[train].mean(0), X[train].std(0) + 1e-6; X = (X - mu) / sd
C = np.stack([df[c].astype(str).map(vocab[c]).fillna(0).to_numpy(np.int64) for c in CATS], 1)
Y = np.stack([df.success.to_numpy(np.float32), df.useful_wip.to_numpy(np.float32), df.useful_star.to_numpy(np.float32)], 1)
val = np.where(df.useful_star.to_numpy(bool), 0.999 ** df.gain_frames.clip(lower=0).to_numpy(np.float64), 0.0).astype(np.float32)
cost = np.log10(df.frames.to_numpy(np.float32)); seg = np.log10(df.seg_frames.clip(lower=1).to_numpy(np.float32)); ok = df.success.to_numpy(bool)
Q = torch.tensor([0.1, 0.5, 0.9], device=dev)


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.ModuleList([nn.Embedding(len(vocab[c]) + 1, 16) for c in CATS])
        d = len(NUM) + 16 * len(CATS); layers = []
        for _ in range(A.layers): layers += [nn.Linear(d, A.hidden), nn.ReLU()]; d = A.hidden
        s.trunk = nn.Sequential(*layers); s.head = nn.Linear(d, 3 + 3 + 3 + 1)

    def forward(s, x, c):
        h = torch.cat([x] + [e(c[:, i]) for i, e in enumerate(s.emb)], 1)
        return s.head(s.trunk(h))


def pinball(pred, y, mask=None):
    e = y[:, None] - pred; l = torch.maximum(Q * e, (Q - 1) * e).sum(1)
    return (l * mask).sum() / mask.sum().clamp(min=1) if mask is not None else l.mean()


t = lambda a: torch.tensor(a, device=dev)
Xt, Ct, Yt, costt, segt, okt, valt = t(X), t(C), t(Y), t(cost), t(seg), t(ok.astype(np.float32)), t(val)
tr_idx = torch.tensor(np.flatnonzero(train), device=dev); te_idx = torch.tensor(np.flatnonzero(test), device=dev)
pos = Yt[tr_idx].mean(0); net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=A.lr, weight_decay=1e-4)
steps = A.epochs * (len(tr_idx) // A.batch + 1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, A.lr, total_steps=steps)


def evaluate():
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 65536):
            j = te_idx[i:i + 65536]; out.append(net(Xt[j], Ct[j]))
    o = torch.cat(out); p = torch.sigmoid(o[:, :3]).cpu().numpy(); yt = Y[test]; m = {}
    for k, n in enumerate(['success', 'pays', 'star']):
        m[n + '_auc'] = roc_auc_score(yt[:, k], p[:, k]); m[n + '_ap'] = average_precision_score(yt[:, k], p[:, k])
    cq = o[:, 3:6].cpu().numpy(); ct = cost[test]
    m['cost_mae_dex'] = float(np.abs(cq[:, 1] - ct).mean()); m['cost_cover_10_90'] = float(((ct >= cq[:, 0]) & (ct <= cq[:, 2])).mean())
    sq = o[:, 6:9].cpu().numpy()[ok[test]]; st = seg[test][ok[test]]
    m['seg_mae_dex'] = float(np.abs(sq[:, 1] - st).mean()); m['seg_cover_10_90'] = float(((st >= sq[:, 0]) & (st <= sq[:, 2])).mean())
    v = torch.sigmoid(o[:, 9]).cpu().numpy(); vt = val[test]
    m['value_mae'] = float(np.abs(v - vt).mean()); m['value_corr_pos'] = float(np.corrcoef(v[vt > 0], vt[vt > 0])[0, 1]) if (vt > 0).sum() > 10 else 0.0
    m['value_auc'] = roc_auc_score(vt > 0, v)
    # What the search would use: paying attempts found in the best 20% of frames, ranked by p(pays) per frame.
    fr = df.frames.to_numpy(float)[test]
    for k, n in [(1, 'pays'), (2, 'star')]:
        o_ = np.argsort(-p[:, k] / fr); c_ = np.cumsum(fr[o_]); kk = np.searchsorted(c_, 0.2 * c_[-1]); m[n + '_top20'] = float(yt[o_, k][:kk].sum() / max(1, yt[:, k].sum()))
    net.train(); return m


with mlflow.start_run(run_name=A.name or f'mlp-{A.split}' + ('-nostate' if A.no_state else '')):
    mlflow.log_params(dict(vars(A), n_train=int(train.sum()), n_test=int(test.sum()), device=dev, params=sum(p.numel() for p in net.parameters())))
    t0 = time.time(); step = 0
    for ep in range(A.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]; tot = 0.0
        for i in range(0, len(perm), A.batch):
            j = perm[i:i + A.batch]; o = net(Xt[j], Ct[j])
            loss = (nn.functional.binary_cross_entropy_with_logits(o[:, :3], Yt[j]) + pinball(o[:, 3:6], costt[j]) + pinball(o[:, 6:9], segt[j], okt[j])
                    + nn.functional.binary_cross_entropy_with_logits(o[:, 9], valt[j]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); step += 1
        m = evaluate() if test.any() and not A.all else {}
        mlflow.log_metric('train_loss', tot / (len(perm) // A.batch + 1), step=ep)
        for k, v in m.items(): mlflow.log_metric(k, v, step=ep)
        print(f'epoch {ep} loss {tot / (len(perm) // A.batch + 1):.4f} ' + ' '.join(f'{k} {v:.3f}' for k, v in m.items()), flush=True)
    mlflow.log_metric('train_secs', time.time() - t0)
    torch.save(net.state_dict(), os.path.join(A.out, 'model.pt'))
    sdict = {k: v.cpu().numpy().tolist() for k, v in net.state_dict().items()}
    json.dump(dict(cats=CATS, vocab=vocab, num=NUM, mu=mu.tolist(), sd=sd.tolist(), layers=A.layers, hidden=A.hidden, heads=['success', 'pays', 'star', 'cost_q10', 'cost_q50', 'cost_q90', 'seg_q10', 'seg_q50', 'seg_q90', 'value'], weights=sdict),
              open(os.path.join(A.out, 'model.json'), 'w'))
    mlflow.log_artifact(os.path.join(A.out, 'model.json'))
