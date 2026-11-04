#!/usr/bin/env python3
"""Overnight: the fastest King Bob-omb kill. Runs forever; safe to kill and restart (state in logs/kingloop/state.json).

Four jobs at a time, each `legs` at 4 threads. Every finished (or dead, or timed-out) job is judged by REPLAYING its
route.m64: the first frame with a star in BoB. A new best per template becomes the seed of later jobs (top-level
reference + OBRRT_GA_KEEP_SEED, so legs still on it can't get worse). No improvement raises the budget.

Job kinds in turn: refine the global best; the same with --window 3; refine the other templates' bests (fresh if they
have none); a fresh run of a template. Results: logs/king_loop.txt, logs/kingloop/best-*.m64, oss2/runs/legs-kingloop-*.
"""
import csv, json, os, shutil, signal, subprocess, sys, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = "logs/kingloop"; STATE = D + "/state.json"; LOG = "logs/king_loop.txt"
LEGS = "builds/legs12/legs"; PLAY = "builds/tools/tasplay"; ROM = "../baserom.us.z64"
TEMPLATES = ["fast-king-blj-tas.ramp0", "fast-king-speed-blj"]
LEVELS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # budget multiplier on population and generations each (cost x its square)
SLOTS = 4; TIMEOUT = 5 * 3600
ENV = dict(os.environ, OBRRT_LAKITU_THROUGH_DOOR="1", OBRRT_GA_KEEP_SEED="1", OBRRT_GA_QD="1",
           OBRRT_GA_GAMMAS="0.98,0.99,0.995,0.999", OBRRT_GA_REGRET="60", OBRRT_KING_LANDING="1", RAYON_NUM_THREADS="4")

def log(*a):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True); open(LOG, "a").write(line + "\n")

def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"n": 0, "since": 0, "best": {}, "history": []}

def save(s):
    t = STATE + ".tmp"; json.dump(s, open(t, "w"), indent=1); os.replace(t, STATE)

def star_frame(m64):
    "The first frame with a star in BoB, from an independent replay; None if there is none."
    if not os.path.getsize(m64): return None
    c = D + f"/replay-{os.getpid()}.csv"
    try:
        subprocess.run([PLAY, ROM, m64, c], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=600)
        for r in csv.DictReader(open(c)):
            if r["level"] == "9" and int(r["stars"]) >= 1:
                return int(r["frame"])
    except Exception as e:
        log("replay failed", m64, e)
    finally:
        if os.path.exists(c): os.remove(c)
    return None

def consider(s, tpl, m64, run):
    f = star_frame(m64) if os.path.exists(m64) else None
    if f is None: return None
    old = s["best"].get(tpl)
    if old is None or f < old["frame"]:
        p = f"{D}/best-{tpl}.m64"; shutil.copy(m64, p + ".tmp"); os.replace(p + ".tmp", p)
        s["best"][tpl] = {"frame": f, "m64": os.path.abspath(p), "run": run}
        log(f"NEW BEST {tpl}: star at frame {f} ({f/30:.2f} s) from {run}")
    return f

def global_best(s):
    return min(s["best"].items(), key=lambda kv: kv[1]["frame"]) if s["best"] else None

def route_file(s, tpl, refine, mult, name):
    d = json.load(open(f"tools/spec/routes/{tpl}.json"))
    b = d.setdefault("budget", {})
    b["population"] = int(b.get("population", 128) * mult); b["generations"] = int(b.get("generations", 60) * mult)
    src = s["best"].get(tpl) or s.setdefault("partial", {}).get(tpl) or {}
    for i, l in enumerate(d["legs"]):
        lb = l.get("budget")
        if lb:
            for k in ("population", "generations"):
                if k in lb: lb[k] = int(lb[k] * mult)
        if refine and i < src.get("legs", 99): l.pop("reference", None)
    if refine:
        d["reference"] = {"m64": src["m64"]}; d["seed_ref"] = True
    d["name"] = name
    p = f"{D}/{name}.json"; json.dump(d, open(p, "w"), indent=1)
    return p

HYPER = "fast-king-blj-tas.ramp0"

def next_job(s):
    "Half the jobs are always the hyperjump route; each template has its own budget level and keeps its solved legs."
    n = s["n"]; s["n"] += 1
    # All slots on the BLJ route. Each slot differs (window, budget level) so they don't repeat each other.
    kind = n % 4; tpl = HYPER if kind in (0, 2) else "fast-king-speed-blj"; window = [1, 1, 2, 3][kind]
    refine = tpl in s["best"] or tpl in s.setdefault("partial", {})
    level = s.setdefault("since_t", {}).get(tpl, 0) // 4 + kind
    mult = LEVELS[min(len(LEVELS) - 1, level)]
    name = f"kingloop-{n:04d}-{'r' if refine else 'f'}{window}-x{mult:g}-{tpl.replace('fast-king-', '').replace('.', '_')}"
    return dict(n=n, tpl=tpl, refine=refine, window=window, mult=mult, name=name)

def note_partial(s, tpl, out, name):
    "A run that didn't reach the star: keep its inputs if it solved more legs (then fewer frames) than before. True if kept."
    try:
        t = json.load(open(out + "/route.json")).get("totals") or {}
        n, fr = int(t.get("legs_solved") or 0), int(t.get("end_frame") or 0)
    except Exception:
        return False
    old = s.setdefault("partial", {}).get(tpl)
    if n == 0 or not os.path.getsize(out + "/route.m64") or (old and (n, -fr) <= (old["legs"], -old["frame"])):
        return False
    p = f"{D}/partial-{tpl}.m64"; shutil.copy(out + "/route.m64", p + ".tmp"); os.replace(p + ".tmp", p)
    s["partial"][tpl] = {"legs": n, "frame": fr, "m64": os.path.abspath(p), "run": name}
    log(f"PARTIAL {tpl}: {n} legs solved, frame {fr}, from {name}")
    return True

def start(s, j):
    out = f"oss2/runs/legs-{j['name']}"; os.makedirs(out, exist_ok=True)
    rf = route_file(s, j["tpl"], j["refine"], j["mult"], j["name"])
    cmd = ["nice", "-n", "5", LEGS, rf, "--out", out, "--beam", "4", "--seed", str(j["n"] + 1)]
    if j["window"] > 1: cmd += ["--window", str(j["window"])]
    lf = open(out + "/legs.log", "w")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=ENV, start_new_session=True)
    log("start", j["name"], "pid", p.pid, "best", {k: v["frame"] for k, v in s["best"].items()})
    return dict(p=p, j=j, out=out, t=time.time())

def main():
    s = load()
    # Seed with what's already there, so refining starts from the best known routes.
    if not s["best"]:
        for tpl, runs in (("fast-king-hits", ["legs-fast-king-hits-s1", "legs-fast-king-hits-s2", "legs-fast-king-hits-m-s1", "legs-fast-king-hits-m-s2"]),):
            for r in runs:
                consider(s, tpl, f"oss2/runs/{r}/route.m64", r)
        save(s)
    running = []
    signal.signal(signal.SIGTERM, lambda *_: ([os.killpg(x["p"].pid, signal.SIGTERM) for x in running], sys.exit(0)))
    while True:
        while len(running) < SLOTS:
            try:
                running.append(start(s, next_job(s))); save(s)
            except Exception as e:
                log("start failed", e); time.sleep(60); break
        time.sleep(30)
        for x in list(running):
            rc = x["p"].poll(); late = time.time() - x["t"] > TIMEOUT
            if rc is None and not late: continue
            if rc is None:
                os.killpg(x["p"].pid, signal.SIGKILL); x["p"].wait(); rc = "timeout"
            running.remove(x); j = x["j"]
            gb0 = global_best(s); before = gb0[1]["frame"] if gb0 else None
            old_t = (s["best"].get(j["tpl"]) or {}).get("frame")
            f = consider(s, j["tpl"], x["out"] + "/route.m64", j["name"])
            better = (s["best"].get(j["tpl"]) or {}).get("frame") != old_t
            if f is None: better = note_partial(s, j["tpl"], x["out"], j["name"])
            st = s.setdefault("since_t", {}); st[j["tpl"]] = 0 if better else st.get(j["tpl"], 0) + 1
            gb1 = global_best(s); after = gb1[1]["frame"] if gb1 else None
            s["since"] = 0 if after != before else s["since"] + 1
            s["history"].append(dict(name=j["name"], rc=rc, star=f, secs=round(time.time() - x["t"])))
            log("end", j["name"], "rc", rc, "star", f, "mins", round((time.time() - x["t"]) / 60), "best", after, "since", s["since"])
            save(s)

if __name__ == "__main__":
    main()
