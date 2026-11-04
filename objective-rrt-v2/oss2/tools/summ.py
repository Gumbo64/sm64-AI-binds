import json, sys, os
S = sys.argv[1]
for name in sys.argv[2:]:
    d = os.path.join(S, "runs", name)
    st = json.load(open(f"{d}/stats.json")); sm = json.load(open(f"{d}/summary.json"))
    ks = stars = 0
    for line in open(f"{d}/cells.jsonl"):
        c = json.loads(line)
        for o, f, v in c["key"]:
            if o == 0 and f == 1: ks = max(ks, v)
            if o == 0 and f == 2: stars = max(stars, v)
    zones = sorted((a["first"], a["zone"] >> 8, a["zone"] & 255, a["n"]) for a in sm["areas"])
    g = st.get("ga", {})
    ops = {k: (v["rollouts"], v["added"], v.get("improved")) for k, v in st["operators"].items() if v["rollouts"]}
    print(f"{name}: cells {st['cells']} frames {st['frames']} {st['frames']/st['elapsed_s']:.0f} f/s replaced {st.get('replaced')} ({st.get('improved_per_mframe',0):.1f}/Mf) keys {ks} stars {stars}")
    print("   zones (first frame, level, area, cells):", zones)
    print("   ops:", ops, "| ga exp", g.get("expansions"), "met", g.get("met"), "fallback", g.get("fallbacks"), "eligible/exp", round(g.get("eligible",0)/max(1,g.get("expansions",1)),1), "frames", g.get("frames"))
    if g.get("kinds"):
        print("   kinds:", {k: (v["tried"], v["met"], v["succ"], v["added"], v["improved"]) for k, v in sorted(g["kinds"].items(), key=lambda kv: -kv[1]["tried"])})
