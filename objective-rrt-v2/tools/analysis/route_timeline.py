#!/usr/bin/env python3
"""Timeline of a run's best route to its highest star count.

    tools/analysis/route_timeline.py runs/<run> [--edges] [--stars N] [--speed U]

The route is the soonest node (by game frame) with the most stars (or N
stars), walked up to the root. Printed:

  1. Edges (--edges): every route node: frame, +frames, zone (level:area),
     stars/coins, `via`, end action and hspeed.
  2. Phases: consecutive edges grouped into castle legs and course visits,
     each split into modal time (Menu::*, Dialog::*, Npc::Talk), star
     grabs, painting entries, exits (Mario::GoToCastle, star kick-out,
     death/other), and the `via` mix of the rest (frames per objective kind).
  3. Re-entries: per castle leg after a star, frames from regaining control
     (the last modal node after arriving) to the painting/course entry; the
     straight-line distance to the painting centre / --speed (default 32,
     Mario's top running speed), the first visit's same leg, and the best
     same-star castle leg anywhere in the tree into that level.
  4. Dwells: per course visit, frames from the last useful event (entry
     menus or a star) to leaving with no star: the `via` mix, the first
     route node with ACT_FLAG_PAUSE_EXIT (when Mario::GoToCastle first
     applied), how many Mario::GoToCastle attempts each visit node got
     (events.jsonl `from`), and the soonest castle arrival in the tree from
     any node of that visit.
  5. Sooner fixed spawns: route states the game puts at a fixed spot (the
     act-select spawn, the exit-course spawn in the castle, the star
     kick-out landing) against the soonest identical one in the tree (level,
     area, act, star flags, keys, 64-unit cell): what grafting the route's
     continuation onto that copy (OBRRT_REROUTE) would save.

Node granularity: nodes.csv has only edge endpoints, so "first possible"
means the first route node whose end state allowed it.
"""
import argparse, collections, csv, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def consts():
    src = open(os.path.join(ROOT, 'core/src/layout.rs')).read()
    acts, levels, c = {}, {}, {}
    for name, val in re.findall(r'pub const (\w+): u32 = (\d+);', src):
        v = int(val)
        c[name] = v
        if name.startswith('ACT_') and not name.startswith(('ACT_FLAG', 'ACT_GROUP')):
            acts.setdefault(v, name[4:].lower())
        if name.startswith('LEVEL_') and name != 'LEVEL_BOUNDARY_MAX':
            levels[v] = name[6:].lower()
    return acts, levels, c


def paintings(levels):
    src = open(os.path.join(ROOT, 'objectives/src/painting_table.rs')).read()
    out = []
    pat = r'name: "(\w+)", area: (\d+), dest: LEVEL_(\w+), pitch: ([-\d.]+), yaw: ([-\d.]+), corner: \[([-\d.]+), ([-\d.]+), ([-\d.]+)\], size: ([-\d.]+)'
    inv = {v: k for k, v in levels.items()}
    for name, area, dest, _pitch, yaw, cx, cy, cz, size in re.findall(pat, src):
        yaw, size = math.radians(float(yaw)), float(size)
        h = size / 2
        centre = (float(cx) + h * math.cos(yaw), float(cy) + h, float(cz) - h * math.sin(yaw))
        out.append(dict(name=name, area=int(area), dest=inv.get(dest.lower()), centre=centre))
    return out


ACTS, LEVELS, C = consts()
PAINTINGS = paintings(LEVELS)
CASTLE = {C['LEVEL_CASTLE'], C['LEVEL_CASTLE_GROUNDS'], C['LEVEL_CASTLE_COURTYARD']}
PAUSE_EXIT = C['ACT_FLAG_PAUSE_EXIT']
MODAL = ('Menu::', 'Dialog::', 'Npc::Talk')


def lvname(l):
    return LEVELS.get(l, str(l))


def actname(a):
    return ACTS.get(a, hex(a))


def load(run):
    nodes = {}
    for r in csv.DictReader(open(os.path.join(run, 'nodes.csv'))):
        n = dict(id=int(r['id']), parent=int(r['parent']), level=int(r['level']), area=int(r['area']),
                 stars=int(r['stars']), coins=int(r['coins']), keys=int(r['keys']), frame=int(r['frame']),
                 via=r['via'], action=int(r['action'], 16), hspeed=float(r['hspeed'] or 0),
                 pos=(float(r['x']), float(r['y']), float(r['z'])), act=r.get('act', ''),
                 flags=r.get('star_flags', ''))
        nodes[n['id']] = n
    events = []
    p = os.path.join(run, 'events.jsonl')
    if os.path.exists(p):
        for line in open(p):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if 'kind' in e and 'from' in e:
                events.append(e)
    return nodes, events


def path_to(nodes, n):
    out = []
    while True:
        out.append(n)
        if n['parent'] < 0:
            break
        n = nodes[n['parent']]
    return out[::-1]


def modal(via):
    return via.startswith(MODAL)


def dist(a, b):
    return math.dist(a, b)


def painting_for(level, pos=None):
    cands = [p for p in PAINTINGS if p['dest'] == level]
    if not cands:
        return None
    if pos is None:
        return cands[0]
    return min(cands, key=lambda p: dist(p['centre'], pos))


def vmix(edges, top=6):
    """Frames per via kind over edges (child nodes), largest first."""
    m = collections.Counter()
    for c, dt in edges:
        m[c['via']] += dt
    return ', '.join(f'{k} {v}' for k, v in m.most_common(top))


def segments(route):
    """Split the route into legs of consecutive nodes in the castle or in one
    course (by level), each a list of route indices."""
    legs, cur = [], [0]
    for i in range(1, len(route)):
        a, b = route[i - 1], route[i]
        same = (a['level'] in CASTLE) == (b['level'] in CASTLE) and (a['level'] in CASTLE or a['level'] == b['level'])
        if same:
            cur.append(i)
        else:
            legs.append(cur)
            cur = [i]
    legs.append(cur)
    return legs


def castle_legs_in_tree(nodes, children):
    """For every node entering a course from the castle (parent in castle),
    the frames since its path first had that star count in the castle
    (the castle arrival after the star) or since the root for 0 stars.
    Returns {(dest level, stars): [(frames, node id, entry frame, arrival pos)]}."""
    first_castle = {}  # node id -> frame its ancestry first stood in the castle at its star count

    def arrival(nid):
        # Iterative: walk up while stars stay the same, record the earliest castle node.
        chain = []
        n = nodes[nid]
        s = n['stars']
        best = None
        while True:
            if n['id'] in first_castle and nodes[n['id']]['stars'] == s:
                best = first_castle[n['id']]
                break
            chain.append(n['id'])
            if n['level'] in CASTLE:
                best = (n['frame'], n['pos'])
            if n['parent'] < 0 or nodes[n['parent']]['stars'] != s:
                break
            n = nodes[n['parent']]
        for c in chain:
            first_castle[c] = best
        return best

    out = collections.defaultdict(list)
    for n in nodes.values():
        if n['parent'] < 0 or n['level'] in CASTLE:
            continue
        p = nodes[n['parent']]
        if p['level'] not in CASTLE or p['stars'] != n['stars']:
            continue
        a = arrival(p['id'])
        if a is None:
            continue
        out[(n['level'], n['stars'])].append((n['frame'] - a[0], n['id'], n['frame'], a[1]))
    return out


def regain(route, leg):
    """Index (into route) of the last modal node at the start of a castle leg
    (the save menu and dialogs after arriving), else the leg's first node."""
    j = leg[0]
    for i in leg:
        if modal(route[i]['via']) or i == leg[0]:
            j = i
        else:
            break
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--edges', action='store_true', help='print every route edge')
    ap.add_argument('--stars', type=int, default=None, help='route to this star count (default: max)')
    ap.add_argument('--speed', type=float, default=32.0, help='units/frame for the straight-line bound')
    a = ap.parse_args()
    nodes, events = load(a.run)
    ms = a.stars if a.stars is not None else max(n['stars'] for n in nodes.values())
    best = min((n for n in nodes.values() if n['stars'] == ms), key=lambda n: (n['frame'], n['id']))
    route = path_to(nodes, best)
    print(f'# {os.path.basename(a.run.rstrip("/"))}: {ms} stars at frame {best["frame"]} (node {best["id"]}, {len(route)} route nodes)')

    att = collections.defaultdict(collections.Counter)  # from node -> kind -> attempts
    for e in events:
        att[e['from']][e['kind']] += 1
    children = collections.defaultdict(list)
    for n in nodes.values():
        if n['parent'] >= 0:
            children[n['parent']].append(n['id'])

    if a.edges:
        print('\n## edges')
        print(f'{"node":>6} {"frame":>6} {"+f":>5} {"zone":>9} {"st":>2} {"co":>3}  via / action hspeed')
        for i, n in enumerate(route):
            dt = n['frame'] - route[i - 1]['frame'] if i else 0
            print(f'{n["id"]:>6} {n["frame"]:>6} {dt:>5} {lvname(n["level"]) + ":" + str(n["area"]):>9} {n["stars"]:>2} {n["coins"]:>3}  {n["via"]} / {actname(n["action"])} {n["hspeed"]:.0f} @{n["pos"][0]:.0f},{n["pos"][1]:.0f},{n["pos"][2]:.0f}')

    legs = segments(route)
    print('\n## phases (frames)')
    reentries, dwells = [], []
    visited = collections.Counter()
    for k, leg in enumerate(legs):
        f0 = route[leg[0] - 1]['frame'] if leg[0] > 0 else route[0]['frame']
        f1 = route[leg[-1]]['frame']
        first, last = route[leg[0]], route[leg[-1]]
        edges = [(route[i], route[i]['frame'] - route[i - 1]['frame']) for i in leg if i > 0]
        modal_f = sum(dt for c, dt in edges if modal(c['via']))
        stars = [(route[i]['frame'], route[i]['via']) for i in leg if i > 0 and route[i]['stars'] > route[i - 1]['stars']]
        if first['level'] in CASTLE:
            nxt = route[leg[-1] + 1] if leg[-1] + 1 < len(route) else None
            r = regain(route, leg)
            rf = route[r]['frame']
            tail = f'-> {lvname(nxt["level"])} via {nxt["via"]} (+{nxt["frame"] - f1})' if nxt else '(end)'
            print(f'castle  {f0:>6}-{f1:<6} {f1 - f0:>5}f  modal {modal_f}  control@{rf}  {tail}')
            move = [(c, dt) for c, dt in edges if not modal(c['via']) and c['frame'] > rf]
            if move:
                print(f'        moves: {vmix(move)}')
            if nxt is not None:
                entry_f = nxt['frame']
                p = painting_for(nxt['level'], route[leg[-1]]['pos'])
                d = dist(route[r]['pos'], p['centre']) if p and route[r]['level'] == C['LEVEL_CASTLE'] else None
                reentries.append(dict(leg=k, stars=route[r]['stars'], dest=nxt['level'], frames=entry_f - rf,
                                      castle_total=entry_f - first['frame'], start=route[r]['pos'], dist=d, pname=p['name'] if p else None,
                                      area=route[r]['area'], regain_level=route[r]['level']))
        else:
            visited[first['level']] += 1
            nxt = route[leg[-1] + 1] if leg[-1] + 1 < len(route) else None
            if nxt is None:
                how = 'end'
            elif nxt['stars'] > last['stars']:
                how = f'star kick-out via {nxt["via"]} (+{nxt["frame"] - f1})'
            elif nxt['via'] == 'Mario::GoToCastle':
                how = f'GoToCastle (+{nxt["frame"] - f1})'
            else:
                how = f'other: {nxt["via"]} / {actname(nxt["action"])} (+{nxt["frame"] - f1})'
            st = ' '.join(f'*@{f}' for f, _ in stars)
            print(f'{lvname(first["level"]):<6}  {f0:>6}-{f1:<6} {f1 - f0:>5}f  modal {modal_f}  {st}  leave: {how}')
            work = [(c, dt) for c, dt in edges if not modal(c['via'])]
            if work:
                print(f'        moves: {vmix(work)}')
            # Dwell: from the last star in the visit (or the entry menus) to leaving, no star.
            if nxt is not None and nxt['stars'] == last['stars']:
                lastuse = leg[0]
                for i in leg:
                    if (i > 0 and route[i]['stars'] > route[i - 1]['stars']) or modal(route[i]['via']) and i <= leg[0] + 3:
                        lastuse = i
                exit_f = nxt['frame']
                span = [i for i in leg if i > lastuse]
                pe = [i for i in [lastuse] + span if route[i]['action'] & PAUSE_EXIT]
                gtc = [(route[i]['id'], att[route[i]['id']]['Mario::GoToCastle'], sum(att[route[i]['id']].values())) for i in [lastuse] + span]
                # Soonest castle arrival in the tree from any node of this visit (same stars).
                soon = None
                for i in [lastuse] + span:
                    for cid in children[route[i]['id']]:
                        c = nodes[cid]
                        if c['level'] in CASTLE and c['stars'] == route[i]['stars']:
                            if soon is None or c['frame'] < soon[0]:
                                soon = (c['frame'], c['via'], route[i]['id'])
                dwells.append(dict(level=first['level'], start=route[lastuse]['frame'], exit=exit_f, how=how,
                                   frames=exit_f - route[lastuse]['frame'],
                                   mix=vmix([(route[i], route[i]['frame'] - route[i - 1]['frame']) for i in span]),
                                   first_pe=(route[pe[0]]['frame'], route[pe[0]]['id'], actname(route[pe[0]]['action'])) if pe else None,
                                   gtc=gtc, soon=soon, next_level=(route[legs[k + 1][-1] + 1]['level'] if k + 2 < len(legs) and legs[k + 1][-1] + 1 < len(route) else None)))

    tree = castle_legs_in_tree(nodes, children)
    print('\n## re-entries (control in castle -> course entry)')
    firsts = {}
    for r in reentries:
        line = f'stars {r["stars"]} -> {lvname(r["dest"])}: {r["frames"]}f from control ({r["castle_total"]}f in castle)'
        if r['dist'] is not None:
            line += f'; straight {r["dist"]:.0f}u = {r["dist"] / a.speed:.0f}f @ {a.speed:g}u/f'
        elif r['regain_level'] != C['LEVEL_CASTLE']:
            line += f'; control in {lvname(r["regain_level"])}'
        prev = firsts.get(r['dest'])
        if prev is not None:
            line += f'; earlier route visit {prev["frames"]}f'
        firsts.setdefault(r['dest'], r)
        alts = tree.get((r['dest'], r['stars']), [])
        if alts:
            bf, bid, bfr, _ = min(alts)
            ef = min(x[2] for x in alts)
            line += f'; tree best castle leg {bf}f (node {bid}), soonest entry @{ef}'
        print(line)

    print('\n## productive re-entries (star arrival -> entry of the next course that gives a star)')
    for k, leg in enumerate(legs):
        first = route[leg[0]]
        if first['level'] not in CASTLE or leg[0] == 0 or first['stars'] <= route[leg[0] - 1]['stars']:
            continue
        r = regain(route, leg)
        # Next course leg whose visit gains a star.
        passes = []
        for k2 in range(k + 1, len(legs)):
            l2 = legs[k2]
            if route[l2[0]]['level'] in CASTLE:
                continue
            nxt = l2[-1] + 1
            gained = any(route[i]['stars'] > route[i - 1]['stars'] for i in l2 if i > 0) or (nxt < len(route) and route[nxt]['stars'] > route[l2[-1]]['stars'])
            if gained:
                break
            passes.append(f'{lvname(route[l2[0]]["level"])} {route[l2[0] - 1]["frame"]}-{route[nxt]["frame"] if nxt < len(route) else "end"}')
        else:
            continue
        e = route[l2[0]]
        p = painting_for(e['level'], route[l2[0] - 1]['pos'])
        line = f'stars {first["stars"]}: arrive @{first["frame"]} control @{route[r]["frame"]} -> {lvname(e["level"])} @{e["frame"]}: {e["frame"] - route[r]["frame"]}f from control'
        if p and route[r]['level'] == C['LEVEL_CASTLE']:
            d = dist(route[r]['pos'], p['centre'])
            line += f'; straight {d:.0f}u = {d / a.speed:.0f}f'
        if passes:
            line += f'; pass-through visits: {", ".join(passes)}'
        alts = tree.get((e['level'], first['stars']), [])
        if alts:
            bf, bid, _, _ = min(alts)
            line += f'; tree best same-star castle leg into {lvname(e["level"])} {bf}f from arrival (node {bid})'
            near = [x for x in alts if dist(x[3], first['pos']) < 1000]
            if near:
                bf, bid, _, _ = min(near)
                line += f', {bf}f from an arrival within 1000u (node {bid})'
            line += f' vs route {e["frame"] - first["frame"]}f'
        print(line)

    print('\n## dwells (in a course, no star, then leave)')
    for d in dwells:
        line = f'{lvname(d["level"])} {d["start"]}->{d["exit"]} {d["frames"]}f, {d["how"]}'
        print(line)
        if d['mix']:
            print(f'    via: {d["mix"]}')
        if d['first_pe']:
            f, nid, an = d['first_pe']
            print(f'    first PAUSE_EXIT route node @{f} (node {nid}, {an}); {d["exit"] - f}f before the castle')
        else:
            print('    no route node with PAUSE_EXIT')
        tried = [f'{nid}:{g}/{t}' for nid, g, t in d['gtc'] if t]
        print(f'    GoToCastle/all attempts per node: {" ".join(tried) or "-"}')
        if d['soon']:
            f, via, frm = d['soon']
            print(f'    soonest castle child in tree @{f} via {via} from node {frm} ({d["exit"] - f:+d}f vs route)')
    sooner_spawns(nodes, route)


def fixed_spawn(n, parent):
    """A state the game puts Mario in at a fixed spot whatever came before:
    the act-select spawn in a course, the exit-course spawn in the castle
    (Mario::GoToCastle), and the star kick-out landing (stars went up)."""
    if n['via'] == 'Menu::SelectAct':
        return 'spawn'
    if n['level'] in CASTLE and n['via'] == 'Mario::GoToCastle':
        return 'exit'
    if parent is not None and n['stars'] > parent['stars'] and n['level'] in CASTLE:
        return 'star'
    return None


def sooner_spawns(nodes, route):
    # Same level, area, act, star flags, keys, 64-unit cell and kind. Coins
    # are left out: they reset on course entry and don't matter in the castle.
    def key(n, kind):
        return (kind, n['level'], n['area'], n['act'], n['flags'], n['keys'], tuple(round(v / 64) for v in n['pos']))
    soonest = {}
    for n in nodes.values():
        kind = fixed_spawn(n, nodes.get(n['parent']))
        if kind:
            k = key(n, kind)
            if k not in soonest or n['frame'] < soonest[k]['frame']:
                soonest[k] = n
    print('\n## sooner fixed spawns (route vs soonest identical state in the tree; what a graft would save)')
    total, run_best = 0, 0
    for i, n in enumerate(route):
        kind = fixed_spawn(n, route[i - 1] if i else None)
        if not kind:
            continue
        b = soonest[key(n, kind)]
        d = n['frame'] - b['frame']
        # Savings don't add up along the route: a later graft point inherits
        # the earlier saving. The route's best single saving is the max of
        # (d - savings already taken) along it; report each and the max.
        run_best = max(run_best, d)
        print(f'{kind:5} {lvname(n["level"])}:{n["area"]} act {n["act"]} stars {n["stars"]}: route @{n["frame"]} (node {n["id"]}), soonest @{b["frame"]} (node {b["id"]}): {d}f')
    print(f'largest single saving {run_best}f')


if __name__ == '__main__':
    main()
