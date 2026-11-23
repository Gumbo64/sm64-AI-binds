#!/usr/bin/env python3
"""Campaign: a 120-star run grown zone by zone, each step a zone-locked tree search (obrrt2) from the route so
far. No reference run is used: the first step starts from power-on with obrrt2's own boot.

    tools/seg120/campaign.py NAME [--threads 16] [--hub 60e6] [--course 150e6] [--grace 20e6] [--steps N]
                             [--bin builds/v2o/obrrt2] [--env "VAR=value ..."] [--keep]

In the castle (inside, grounds, courtyard: one zone) the tree runs until it enters a course it hasn't given up
on. In a course it runs until an exit with more stars (or keys), plus a grace for a faster one, and the route
takes the exit with the fewest frames per star. A course that gives nothing in its budget is skipped until the
star count changes; when every course the castle offers is skipped, budgets double and the list is cleared.
Checkpoints (--checkpoints N, default 8 in a row): a course tree that ends with no exit but got further along one
of its goals (summary.json's `goals`: the grounded state furthest along each; another level such as a Bowser
arena, more work in progress, or an ETA a fifth lower) is continued by a fresh tree from that state instead of
being given up. A course given up after checkpoints is resumed from the last one while stars and keys are the same.
State in logs/campaign/NAME/ (state.json, log.txt, route-<step>.m64); safe to stop and restart."""
import argparse, collections, csv, json, os, shutil, subprocess, sys, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
csv.field_size_limit(10**8)
ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('--threads', type=int, default=16)
ap.add_argument('--hub', type=float, default=60e6); ap.add_argument('--course', type=float, default=150e6)
ap.add_argument('--grace', type=float, default=20e6); ap.add_argument('--steps', type=int, default=10**6)
ap.add_argument('--bin', default='builds/v2o/obrrt2'); ap.add_argument('--keep', action='store_true')
ap.add_argument('--env', default='', help='extra settings for every tree, "VAR=value VAR=value"')
ap.add_argument('--stall', type=float, default=0, help='course trees stop after this many frames without new work in progress (OBRRT_STOP_STALL)')
ap.add_argument('--checkpoints', type=int, default=8, help='continue a course tree that found no exit from its furthest state, at most this many times in a row (0: never)')
ap.add_argument('--coins', type=float, default=0, help='coin legs toward a course\'s missing 100-coin star: frames per leg (0: off)')
ap.add_argument('--coin-retries', type=int, default=3, help='weak coin legs in a row retried (each with twice the frames) before the legs end')
ap.add_argument('--coin-legs', type=int, default=20, help='at most this many coin legs per course visit')
ap.add_argument('--coin-area-move', type=int, default=1, help='a coin leg that gains nothing goes on into another area that holds the coins left for 100 (coin_inventory.json)')
ap.add_argument('--coin-gain', type=int, default=4, help='two coin legs in a row gaining fewer coins than this end the legs')
A = ap.parse_args()
D = f'logs/campaign/{A.name}'; os.makedirs(D, exist_ok=True)
HUB = {6, 16, 26}
LV = {9:'BoB',24:'WF',12:'JRB',5:'CCM',4:'BBH',7:'HMC',22:'LLL',8:'SSL',23:'DDD',10:'SL',11:'WDW',36:'TTM',13:'THI',14:'TTC',15:'RR',27:'PSS',6:'Castle',16:'Grounds',26:'Courtyard',29:'TotWC',28:'CotMC',18:'VCutM',31:'WMotR',20:'SA',17:'BitDW',19:'BitFS',21:'BitS',30:'Bowser1',33:'Bowser2',34:'Bowser3'}
ARENA = {17: 30, 19: 33, 21: 34}
# Level -> course, and how many stars a course holds (the Bowser courses also hold a key).
COURSE = {9:1,24:2,12:3,5:4,4:5,7:6,22:7,8:8,23:9,10:10,11:11,36:12,13:13,14:14,15:15,17:16,19:17,21:18,27:19,28:20,29:21,18:22,31:23,20:24}
STARS_IN = {c: 7 for c in range(1, 16)} | {16: 1, 17: 1, 18: 1, 19: 2, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1}
KEY_OF = {17: 1, 19: 2}
# Cap-switch courses: the save flag their switch sets (nodes.csv `caps`). The star there ends the visit, so a
# visit with the switch unpressed only counts when it leaves with the switch pressed.
CAP = {29: 2, 28: 4, 18: 8}
# A cap course entered from another course, not from the castle (the Cavern of the Metal Cap, from HMC): the
# castle goes to that course, and its tree takes an entry into the cap course as an exit worth having.
VIA = {28: 7}

def done_levels(flags_text, keys, caps=0):
    '''Levels with nothing left to give: every star of the course had, its key if it has one, its cap switch pressed.'''
    f = flags(flags_text) if flags_text else {}
    return sorted(l for l, c in COURSE.items() if bin(f.get(c, 0)).count('1') >= STARS_IN[c] and keys >= KEY_OF.get(l, 0) and caps & CAP.get(l, 0) == CAP.get(l, 0))

def no_go(s):
    '''Levels the castle doesn't go to: the skip list, courses with nothing left (done_levels), and a course whose
    only missing star is its 100 coins after coin legs gave up there (coin_skip): a plain tree can't get it. t1
    steps 550-554, 609 and 622: JRB at 0xbf, picked again and again. Both the tree's OBRRT_EXIT_SKIP and the
    campaign's choice of exit use it (step 622 took JRB's painting with only the first).'''
    f = flags(s['flags']) if s['flags'] else {}
    return sorted(set(s['skip']) | set(done_levels(s['flags'], s['keys'], s.get('caps', 0)))
                  | {lv for lv in s.get('coin_skip', []) if lv in COURSE and f.get(COURSE[lv], 0) == 0x3f})

RACES = (5, 13)    # CCM (the big penguin), THI (Koopa the Quick)
# Courses whose objectives (builds/v4m) steer by walking distance: the races and THI's Secrets::Tour (5, 13), SSL
# and SL (walk.rs: reach, star and secret ETAs), WDW (WdwWater::Lower, WdwRide), HMC (red coins). The walkable mesh
# (OBRRT_GEO=1) is only built where it is on.
# Courses whose coins are all in one area (so the probe's coin inventory is complete): BoB, WF, JRB, BBH, HMC, TTC, RR.
SINGLE_AREA = (9, 24, 12, 4, 7, 14, 15)
GEO_LEVELS = RACES + (8, 11, 7, 4)   # + BBH (mansion rooms, builds/v4n); not SL: step 696 ran at 32-37 k frames/s with it (serial planning), SL's walking ETAs were never tested
BITS = 21          # Bowser in the Sky; its arena (Bowser 3) is 34


def final(s):
    '''All 120 stars and both keys: the last search beats Bowser 3 and touches the grand star.'''
    return s['stars'] >= 120 and s['keys'] >= 2


def want_target(s):
    '''Where the castle must go before anywhere else: Bowser in the Sky at the end (it counts as done once its
    red-coin star is had), else a cap course with its switch unpressed (want_cap).'''
    return BITS if final(s) else (want_cap(s) or want_open(s))


# A course never entered from the castle, and one entered from the same castle area: its hops are used (Rainbow
# Ride's painting is upstairs with Tick Tock Clock's).
NEAR = {15: 14}
# A course's way in from the castle, from the game's level files, for a course the route never entered: Rainbow
# Ride's is the SURFACE_WOBBLING_WARP quad (painting warp node 0x2A) at y 2816 in castle area 2
# (levels/castle_inside/areas/2/collision.inc.c, script.c), the centre of its 4 corners.
ENTRY_HINT = {15: ((6, 2), (-3410.0, 2816.0, 5886.0))}


def want_open(s):
    '''After a castle tree found no new course (budgets doubled, the skip list cleared): the main course with the
    most stars missing, the least visited first, not one with only its 100 coins left after coin legs gave up.
    t1 at 85 stars: Rainbow Ride (0 of 7) never entered and Snowman's Land (1 of 7) once in 623 steps, while the
    castle's trees took the nearest paintings (JRB 21 times, WF 16, BoB 15).'''
    # Every castle step, not only after one found nothing (from step 654): the castle's trees otherwise take the
    # nearest paintings, and courses with the most stars left (RR 6, SL 6 at 86 stars) waited behind
    # courses whose last stars no objective gets (WDW's elevator, THI's Wiggler).
    f = flags(s['flags']) if s['flags'] else {}
    been = collections.Counter(h.get('to') for h in s['history'])
    open_ = [l for l, c in COURSE.items() if 1 <= c <= 15 and l not in no_go(s) and l not in s.get('open_fail', [])]
    if not open_: return None
    return max(open_, key=lambda l: (STARS_IN[COURSE[l]] - bin(f.get(COURSE[l], 0)).count('1'), -been[LV[l]]))


def want_cap(s):
    '''A cap-switch course the campaign has been in whose switch is not pressed (and not given up on): the castle
    goes there before anywhere else. A cap is a precondition of stars in other courses.'''
    been = {h.get('to') for h in s['history']}
    for l, bit in CAP.items():
        if not s.get('caps', 0) & bit and LV[l] in been and l not in s['skip']: return l
    return None
ENV = ("OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=6000 OBRRT_ROUND=64 OBRRT_BOOT_NOWAIT=1 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 "
       "OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_CHAIN=route OBRRT_SAMPLER=rate "
       "OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen "
       "OBRRT_KINSHIP_MULT=10 OBRRT_ZONE_LOCK=1").split()

def log(*a):
    line = time.strftime('%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
    print(line, flush=True); open(f'{D}/log.txt', 'a').write(line + '\n')

def load():
    p = f'{D}/state.json'
    if os.path.exists(p): return json.load(open(p))
    return dict(step=0, route=None, hub_route=None, level=None, stars=0, keys=0, flags='', skip=[], mult=1.0, frames=0, history=[])

def save(s):
    json.dump(s, open(f'{D}/state.json.tmp', 'w'), indent=1); os.replace(f'{D}/state.json.tmp', f'{D}/state.json')

def flags(t):
    o = {}
    for p in t.split():
        c, m = p.split(':'); o[int(c)] = int(m, 16) & 0x7f
    return o

def new_stars(a, b):
    fa, fb = flags(a), flags(b)
    return sorted((c, i + 1) for c, m in fb.items() for i in range(7) if (m & ~fa.get(c, 0)) >> i & 1)

def dead_ends(s, zone):
    '''Where this course's earlier checkpoints were taken, with the same stars had (they led to no exit, or the
    stars would differ): (level, pos) pairs the next checkpoint keeps away from.'''
    seen = s.get('cp_seen', {}).get(str(zone))
    return [(l, q) for l, q in seen['at']] if seen and same_course(seen['flags'], s['flags'], zone) else []


def same_course(a, b, zone):
    '''Whether two star-flag texts agree on this course's stars. Only those: a star elsewhere doesn't reopen a dead
    end here (t1 steps 590 and 613: the same BBH checkpoint at (1000, 0, 1000), the memory cleared by the BitS and
    Koopa race stars in between).'''
    c = COURSE.get(zone)
    return a == b if c is None else flags(a).get(c, 0) == flags(b).get(c, 0)


def cannon_opened(out, root):
    '''A state of the tree that has opened a cannon the root hadn't (the Bob-omb Buddy's flag, bit 0x80 of the NEXT
    course's star byte: save_file_set_cannon_unlocked indexes courseStars[gCurrCourseNum]), grounded, idle or
    walking, with health; the earliest. The checkpoint goals don't count it: t1 step 756 opened RR's cannon in 72
    states and checkpoint 2 went to a box without it; the cannon-unlocked state, put in by hand, gave the box star
    in 95.6 M frames (step 757).'''
    def cannons(t):
        return {c for c, m in flags_raw(t).items() if m & 0x80}
    have = cannons(root['star_flags'])
    best = None
    for r in csv.DictReader(open(out + '/nodes.csv')):
        if not (cannons(r['star_flags']) - have): continue
        if int(r['action'], 16) not in (0x0c400201, 0x04000440, 0x0c400202) or int(r['health'], 16) < 0x500: continue
        if best is None or int(r['frame']) < int(best['frame']): best = r
    return best


def cannon_cp(out, root):
    '''`cannon_opened` as a checkpoint record (the fields the caller logs and keeps).'''
    r = cannon_opened(out, root)
    if r is None: return None
    pos = [float(r['x']), float(r['y']), float(r['z'])]
    return dict(node=int(r['id']), kind='Cannon::Opened', key=None, how='moved', frame=int(r['frame']), level=int(r['level']),
                pos=pos, eta=0.0, eta0=None, wip=0, wip0=0, weight=0.0)


# One-shot steps of a star's chain whose success is worth restarting from when the tree found no exit: the
# checkpoint goals don't rank them (a done step has no ETA left). t1 step 812: HMC pressed the grill switch 174 times
# in 400 M frames, checkpoint 1 went to a 3-red-coin state; the earliest switch state, put in by hand, gave the
# metal-head star in 28.8 M frames (step 813). Eyerok (step 790) and RR's cannon (step 756) were the same story.
STAGE_KINDS = ('Hmc::MetalSwitch', 'Pyramid::TopHole', 'WarpFloor::Enter', 'PenguinRace::Win', 'Snowman::LeadBody', 'WdwWater::Slab', 'MrI::Box', 'MrI::Climb', 'MrI::Gate')


def stage_cp(out, root, zone, avoid=(), used=()):
    '''The earliest healthy state (5 or more health segments, not hurt, not airborne or in a cutscene) that a
    STAGE_KINDS objective's success added, in the course's level, not near an earlier dead checkpoint; as a
    checkpoint record.'''
    added = {}
    try:
        for l in open(out + '/events.jsonl'):
            if not any(k in l for k in STAGE_KINDS): continue
            e = json.loads(l)
            # A kind once per course visit (`used`): DDD's warp floors go both ways (t1 step 827's checkpoint was a
            # WarpFloor::Enter into area 2; the next tree's could be the one back).
            if e.get('kind') in STAGE_KINDS and e.get('kind') not in used and e.get('success'):
                a = e.get('added'); a = a if isinstance(a, list) else [a]
                for n in a:
                    if n is not None: added[str(n)] = e['kind']
    except (OSError, ValueError): return None
    if not added: return None
    best = None
    for r in csv.DictReader(open(out + '/nodes.csv')):
        if r['id'] not in added or int(r['frame']) <= int(root['frame']) or int(r['level']) != zone: continue
        a = int(r['action'], 16)
        if a & 0x1C0 not in (0x000, 0x040, 0x0C0) or a & 0x20000 or int(r.get('health') or '0x880', 16) < 0x500: continue
        pos = [float(r['x']), float(r['y']), float(r['z'])]
        if any(int(r['level']) == l and sum((x - y) ** 2 for x, y in zip(pos, q)) < 300 ** 2 for l, q in avoid): continue
        if best is None or int(r['frame']) < int(best['frame']): best = r
    if best is None: return None
    pos = [float(best['x']), float(best['y']), float(best['z'])]
    return dict(node=int(best['id']), kind=added[best['id']], key=None, how='stage', frame=int(best['frame']), level=int(best['level']),
                pos=pos, eta=0.0, eta0=None, wip=0, wip0=0, weight=0.0)


def flags_raw(t):
    o = {}
    for p in t.split():
        if ':' in p:
            c, m = p.split(':'); o[int(c)] = int(m, 16)
    return o


# Objective kinds that are boss fights (their wip is the boss's lost health): see checkpoint().
BOSS_KINDS = {'Eyerok', 'BigBoo', 'Bowser', 'Wiggler', 'WhompKing', 'KingBobomb', 'BigBully', 'Bully', 'MrI', 'Chuckya', 'Klepto'}


def checkpoint(out, root, zone, n, prefer=None, avoid=(), keys=0, end=False):
    """The state to continue a course from when its tree found no exit. summary.json's `goals` lists, per goal and
    per amount of work in progress, the grounded state with the least ETA. A state in another level (an arena) comes
    first. Then the goal the last checkpoint was taken for (`prefer`, its key), while it shows progress again: a
    climb is kept up. Then two kinds of progress take turns (n = checkpoints so far in a row), each falling back to
    the other: ETA progress, the goal with the most weight x frames of ETA gained against the root's (or, for a goal
    in another area, against the first state that had one), at least 100 frames and a twentieth, taking the state
    with the most work in progress among those nearly as far; and more work in progress than the root (red coins,
    boss damage), the least ETA among the most. None without either."""
    try: goals = json.load(open(out + '/summary.json')).get('goals', [])
    except Exception: return None
    M = E = W = P = B = None
    for g in goals:
        # Movement-only goals (Tech::*) may lead the first checkpoint, not later ones: t1 BBH checkpoints 1 and 3
        # (steps 589-592) and CCM checkpoints 2 and 3 (steps 604-605) went for Tech::HyperWalk, 300 units apart,
        # 800 M frames each, no star.
        if n >= 1 and str(g.get('kind', '')).startswith('Tech::'): continue
        cps = g.get('cps') or [g]   # builds before v3j: one state per goal
        cps = [c for c in cps if c['frame'] > int(root['frame'])]
        # Not where an earlier checkpoint of this course already led nowhere (within 300 units): WDW's star 4
        # in the elevator's cage rated 'ETA 6' from the cage roof right above it, and three trees from there
        # (steps 211, 365, 366; 1.2 B frames) never got in.
        cps = [c for c in cps if not any(c['level'] == l and sum((a - b) ** 2 for a, b in zip(c['pos'], q)) < 300 ** 2 for l, q in avoid)]
        if not cps: continue
        base = g['eta0'] if g.get('eta0') is not None else g.get('eta1')
        flat = lambda c, how: dict(c, kind=g['kind'], key=g.get('key'), weight=g['weight'], eta0=base, wip0=g['wip0'], how=how)
        for c in cps:
            # Only an arena still worth a fight: its key not had (Bowser 3: only at the end). t1 step 470: BitDW at 7
            # of 8 red coins, checkpoint 2 went to the Bowser 1 arena (key 1 had since step 147), no star, skipped.
            if c['level'] != zone and {30: keys < 1, 33: keys < 2, 34: end}.get(c['level'], True):
                k = (g['weight'], c['wip'], -c['eta'])
                if M is None or k > M[0]: M = (k, flat(c, 'moved'))
        e = w = None
        emin = min(c['eta'] for c in cps)
        if base is not None and base - emin > max(100, 0.05 * base):
            c = max((c for c in cps if c['eta'] <= emin + 0.1 * (base - emin)), key=lambda c: (c['wip'], -c['eta']))
            e = ((g['weight'] * (base - emin),), flat(c, 'eta'))
            if E is None or e[0] > E[0]: E = e
        c = max(cps, key=lambda c: (c['wip'], -c['eta']))
        if c['wip'] > g['wip0']:
            w = ((c['wip'], g['weight'], -c['eta']), flat(c, 'wip'))
            if W is None or w[0] > W[0]: W = w
            # A boss half fought comes before any ETA-only progress: t1 step 790's checkpoint 3 went to a
            # Secrets::Tour state (wip 3 -> 3) over Eyerok with one hand dying (wip 7); from that Eyerok state,
            # put in by hand, the star came in 27.7 M frames (step 791).
            if str(g.get('kind', '')).split('::')[0] in BOSS_KINDS and (B is None or w[0] > B[0]): B = w
        if prefer is not None and g.get('key') == prefer[0]:
            P = (e or w) if prefer[1] == 'eta' else (w or e)
    for c in [M, B, P] + ([E, W] if n % 2 == 0 else [W, E]):
        if c is not None: return c[1]
    return None

# The castle's own stars, by save flag bit: the castle area, and the stars and keys the route needs first (Toad's
# in the basement, upstairs, and on the third floor behind the 50-star door; MIPS's twice in the basement).
CASTLE_STARS = {24: ((6, 3), 12, 1), 25: ((6, 2), 25, 2), 26: ((6, 2), 50, 2), 27: ((6, 3), 15, 1), 28: ((6, 3), 50, 1)}

# Where Mario was in the castle, per level, the last time the route entered it from there (a painting, a hole):
# filled by `replay`, the last hop of `zone_path`.
ENTRY_AT = {}


def replay(route):
    """What a replay of the route's own inputs shows: its changes between castle areas ({from: {to: (Mario's place
    at the change, True)}}; a door the route went through one way only is taken to work the other way too, from
    where Mario came out, marked False), the castle area each level was entered from, and the save flags at the end."""
    edges, entry, flags_ = {}, {}, 0
    if not route: return edges, entry, flags_
    tmp = f'{D}/zones.csv'
    subprocess.run(['nice', '-n', '5', 'builds/tools/tasplay', '../baserom.us.z64', route, tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(tmp): return edges, entry, flags_
    prev = at = None
    for r in csv.DictReader(open(tmp)):
        z, pos = (int(r['level']), int(r['area'])), tuple(float(r[k]) for k in 'xyz')
        if prev and z != prev and prev[0] in HUB:
            if z[0] in HUB:
                edges.setdefault(prev, {})[z] = (at, True)
                if not edges.setdefault(z, {}).get(prev, (None, False))[1]: edges[z][prev] = (pos, False)
            else: entry[z[0]] = prev; ENTRY_AT[z[0]] = at
        prev, at, flags_ = z, pos, int(r['save_flags'], 16)
    os.remove(tmp)
    return edges, entry, flags_

def hops(edges, to):
    """Hops between castle areas toward area `to`, as OBRRT_ZONE_PATH text (objectives/src/extra/warps.rs Zone::Go):
    'L.A>L.A@x,y,z;...', one per castle area that isn't `to`, with the place of that change (see `replay`)."""
    if to is None: return ''
    nxt, todo = {}, [to]
    while todo:
        b = todo.pop(0)
        for a, e in edges.items():
            if b in e and a != to and a not in nxt: nxt[a] = b; todo.append(a)
    return ';'.join(f'{a[0]}.{a[1]}>{b[0]}.{b[1]}@' + ','.join(f'{v:.0f}' for v in edges[a][b][0]) for a, b in sorted(nxt.items()))

def zone_path(route, want):
    """Hops toward the castle area the route once entered level `want` from (or a course near it, NEAR); '' when
    it never did from the castle (the metal cap cavern is entered from HMC)."""
    edges, entry, _ = replay(route)
    path = hops(edges, entry.get(want, entry.get(NEAR.get(want))))
    # And the way in from that area, where the route once went in: a castle tree that wanted Snowman's Land got
    # upstairs (6.2) and stayed there, 300 M frames (t1 step 626; Rainbow Ride, step 625, the same).
    if want in entry and want in ENTRY_AT: a, at = entry[want], ENTRY_AT[want]
    elif want in ENTRY_HINT: a, at = ENTRY_HINT[want]
    else: return path
    return ';'.join(x for x in (path, f'{a[0]}.{a[1]}>{want}.1@' + ','.join(f'{v:.0f}' for v in at)) if x)

def castle_star(s):
    """A castle star the route can have by now and hasn't (CASTLE_STARS, the save flags at the route's end), unless
    a tree failed at it with these stars and keys: (its castle area, the hops there). The castle's trees stop at
    the first course entered, so its own stars were only had by chance (MIPS's once in 173 steps of t1)."""
    # After a failed try, again only with 5 more stars or another key: t1 steps 188 and 195 each spent 150 M on
    # Toad's upstairs star (the tree got upstairs, 1093 of 29813 nodes) at 34 and 35 stars.
    t = s.get('castle_try')
    if t and s['keys'] == t[1] and s['stars'] < t[0] + 5: return None
    edges, _, fl = replay(s['route'])
    for bit, (z, n, k) in sorted(CASTLE_STARS.items()):
        if not fl >> bit & 1 and s['stars'] >= n and s['keys'] >= k: return z, hops(edges, z)
    return None

MOAT = 0x200       # SAVE_FLAG_MOAT_DRAINED: both basement pillars pounded (water_pillar.inc.c)


def moat_step(s):
    '''The moat to drain before the vanish cap course can be reached: DDD's first star had (the pillars' room), the
    vanish switch not pressed, the route's save flags without the drain, and no failed try at these stars and keys.
    Returns the hops to the basement (castle area 3), possibly ''.'''
    if s.get('moat_try') == [s['stars'], s['keys']] or s.get('caps', 0) & 0x8: return None
    if not flags(s['flags']).get(9, 0) & 1: return None
    edges, _, fl = replay(s['route'])
    if fl & MOAT: return None
    return hops(edges, (6, 3))


def pause_exit(route, out):
    '''Leave the course from the pause menu: START, the stick down to "Exit Course", A (raw inputs), then idle until
    Mario is in control in the castle. Checked by replay (tasplay): written to `out` up to that frame, with
    (level, stars, frame), only if it gets there with the stars the route had. t1 step 463: BitFS's red-coin star
    had without leaving, and no way out found by a tree (its one star held, the arena a dead end).'''
    import struct, tempfile
    pad = lambda b=0, y=0: struct.pack('>Hbb', b, 0, y)
    base = open(route, 'rb').read(); base = base[0x400:] if base[:4] == b'M64\x1a' else base
    n0 = len(base) // 4
    tail = pad(0x1000) + pad() * 30 + pad(0, -80) * 3 + pad() * 10 + pad(0x8000) + pad() * 400
    with tempfile.TemporaryDirectory() as tmp:
        open(f'{tmp}/m.m64', 'wb').write(base + tail)
        subprocess.run(['nice', '-n', '10', 'builds/tools/tasplay', '../baserom.us.z64', f'{tmp}/m.m64', f'{tmp}/p.csv'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows = list(csv.DictReader(open(f'{tmp}/p.csv'))) if os.path.exists(f'{tmp}/p.csv') else []
    if len(rows) < n0: return None
    stars0 = int(rows[n0 - 1]['stars'])
    for r in rows[n0:]:
        if int(r['level']) in HUB and r['controllable'] == '1' and r['play_mode'] == '0':
            if int(r['stars']) < stars0: return None
            f = int(r['frame'])
            open(out, 'wb').write(base + tail[:(f - n0) * 4])
            return int(r['level']), int(r['stars']), f
    return None


COIN_INV = None
COIN_WARPS = {}


def coin_inventory():
    '''tools/seg120/coin_inventory.json (coin_inventory.py: the coins each area of each course holds, from the level
    files), {course: {area: coins}} with the coins an enemy drops included; {} when the file is missing.'''
    global COIN_INV, COIN_WARPS
    if COIN_INV is None:
        try:
            raw = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coin_inventory.json')))
            COIN_INV = {int(c): {int(a): v['total'] for a, v in d['areas'].items()} for c, d in raw.items()}
            COIN_WARPS = {int(c): {int(a): v['warps'] for a, v in d['areas'].items()} for c, d in raw.items()}
        except (OSError, ValueError, KeyError):
            COIN_INV = {}
    return COIN_INV


def reachable_areas(level, area):
    '''Areas of the course Mario can get to from `area` by the level's own warps and instant warps (the pyramid has no
    way back out: SSL's area 2 only leads on to Eyerok's room, so the coins left outside are lost once he is in).'''
    coin_inventory(); w = COIN_WARPS.get(COURSE.get(level), {}); seen = {area}; todo = [area]
    while todo:
        for b in w.get(todo.pop(), []):
            if b not in seen: seen.add(b); todo.append(b)
    return seen


def other_areas_left(level, area, cm):
    '''Coins the course's areas other than `area` still hold: their inventory less what the legs took there
    (cm['got'][area], the coins a leg gained, booked to the area the leg began in). The coin counter is per level
    visit and a coin taken stays taken when its area is left (numCoins is reset only on entering the level, level_update.c;
    coin.inc.c sets the respawn bits), so the coins that count toward 100 are the course's, not the area's. The loaded
    objects (`area_has_coins`) only show Mario's own area: SSL (t1 steps 776-787) stopped at 72 "too few coins" in the
    pyramid, which the outside's 72 and the pyramid's 59 would have made 131.'''
    inv = coin_inventory().get(COURSE.get(level), {})
    got = (cm or {}).get('got', {})
    ok = reachable_areas(level, area)
    return {a: max(0, n - int(got.get(str(a), 0))) for a, n in inv.items() if a != area and a in ok}


def cm_with(cm, root, node):
    '''`cm` with the coins the leg to `node` gained booked to the leg's own area (root's), for the checks made before the
    leg is booked.'''
    got = dict(cm.get('got', {})); a = str(int(root['area']))
    got[a] = int(got.get(a, 0)) + max(0, int(node['coins']) - int(root['coins']))
    return dict(cm, got=got)


def area_has_coins(out, node, cm=None):
    '''Whether the coins collectable in `node`'s area (loose coins: red 2, blue 5; unspawned coin formations: 5
    or 8) and its coins reach 100: the state replayed with builds/tools/probe. True when it can't be checked.'''
    import re, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        m = f'{tmp}/n.m64'
        if subprocess.run(['builds/tools/nodepath', out, node['id'], m], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode: return True
        pr = subprocess.run(['nice', '-n', '10', 'builds/tools/probe', '../baserom.us.z64', m], capture_output=True, text=True)
    if pr.returncode: return True
    left = 0
    for l in pr.stdout.split('\n')[1:]:
        x = re.match(r'\s*\d+ (\S+).*?interact\s+(0x[0-9a-f]+) act\s+(\d+).*?params (0x[0-9a-f]+)', l)
        if not x: continue
        b, it, act, par = x[1], int(x[2], 16), int(x[3]), int(x[4], 16)
        # Blue coins hidden behind their switch (5 each, DDD's area 2 had six: 'too few coins' at 44 with 30 of them), a
        # box's coins (type 4: 1, 5: 3, 6: 10: exclamation_box.inc.c) and what enemies drop (oNumLootCoins, the probe's `loot`).
        if it == 0x10: left += 2 if b == 'bhvRedCoin' else 5 if 'Blue' in b else 1
        if b == 'bhvExclamationBox' and act <= 2 and (par >> 16) & 0xFF in (4, 5, 6): left += {4: 1, 5: 3, 6: 10}[(par >> 16) & 0xFF]
        lt = re.search(r'loot\s+(\d+)', l)
        if lt and it != 0x10: left += int(lt[1])
        if b == 'bhvCoinFormation' and act == 0: left += 5 if (par >> 16) & 7 < 2 else 8
    # The other areas' coins (inventory), and for the node's own area at least what the inventory says it still holds
    # (the probe doesn't see enemies that aren't spawned, or loot).
    inv = coin_inventory().get(COURSE.get(int(node['level'])), {})
    area = int(node['area'])
    if area in inv: left = max(left, inv[area] - int((cm or {}).get('got', {}).get(str(area), 0)))
    left += sum(other_areas_left(int(node['level']), area, cm).values())
    return int(node['coins']) + left >= 100


def coin_potential(out, node):
    '''Coins `node` has plus the coins its area still offers (twostar.rs `coin_sources`: loose coins, formations, boxes,
    a blue coin switch's, enemy loot), from `probe` with OBRRT_TWOSTAR=1 (its "coins N sources M worth W" line); None
    when it can't be checked.'''
    import re, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        m = f'{tmp}/n.m64'
        if subprocess.run(['builds/tools/nodepath', out, node['id'], m], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode: return None
        pr = subprocess.run(['nice', '-n', '10', 'builds/tools/probe', '../baserom.us.z64', m], capture_output=True, text=True, env=dict(os.environ, OBRRT_TWOSTAR='1'))
    x = re.search(r'^coins (\d+) sources \d+ worth (-?[\d.]+)', pr.stdout, re.M) if pr.returncode == 0 else None
    return int(x[1]) + int(float(x[2])) if x else None


def pick_feasible(out, inside, single_area):
    '''The leg's state to go on from: the most coins among those that can still reach 100 (the top 8 distinct coin
    counts checked, `coin_potential`), else the most coins. A coin leg chose the most coins alone, and JRB's (t1
    steps 539-547) went on from 83 -> 94 coins after a blue coin switch had been left with 10 of its coins: no state
    of the tree could reach the star any more. Only for courses with one area (the coins of the others aren't loaded).'''
    best = max(inside, key=lambda r: (int(r['coins']), -int(r['frame'])), default=None)
    if best is None or not single_area: return best
    seen = set()
    for r in sorted(inside, key=lambda r: (-int(r['coins']), int(r['frame']))):
        if int(r['coins']) in seen: continue
        seen.add(int(r['coins']))
        if len(seen) > 8: break
        pot = coin_potential(out, r)
        if pot is None: return best
        if pot >= 100 - 3:
            if r is not best: log(f'coin leg: from {r["coins"]} coins (potential {pot}), not from {best["coins"]}: that one cannot reach 100')
            return r
    return best


def run_tree(s, budget, grace, star=None, coins=False, moat=None):
    out = f'runs/camp-{A.name}-{s["step"]:04d}'; shutil.rmtree(out, ignore_errors=True)
    env = dict(os.environ, RAYON_NUM_THREADS=str(A.threads), OBRRT_MAX_FRAMES=str(int(budget)), OBRRT_STOP_EXIT_GRACE=str(int(grace)))
    env.update(e.split('=', 1) for e in ENV + A.env.split())
    skip = no_go(s)
    # The castle is rooms and corridors: goals ranked and approached along the walkable mesh (core/src/geo.rs).
    # zt-totwc-4 against zt-totwc-3 (2026-11-10, basement to the Tower of the Wing Cap): 15.9 M frames against none in 41.4 M.
    if s['level'] is None or s['level'] in HUB: env.setdefault('OBRRT_GEO', '1')
    # Races: their ETA and steering by walking distance (builds/v4i+, npcs.rs) only with the walkable mesh on. The
    # Koopa race was won so (zt-ktq-geo-1: 293 of 345, against 0 of 805 without); CCM's penguin race is left.
    if s['level'] in GEO_LEVELS: env.setdefault('OBRRT_GEO', '1')
    if moat is not None:
        # The moat: no course is an exit; the tree stops on the first state in control with the drain flag.
        skip = sorted(COURSE); env['OBRRT_STOP_FLAG'] = hex(MOAT)
        if moat: env['OBRRT_ZONE_PATH'] = moat
        log(f'step {s["step"]}: drain the moat (basement pillars)' + (f' by {moat}' if moat else ''))
    elif star:
        # A castle star: no course is an exit, the tree stops on the first state with more than the root has.
        skip = sorted(COURSE); env['OBRRT_STOP_PROGRESS'] = '1'
        if star[1]: env['OBRRT_ZONE_PATH'] = star[1]
        log(f'step {s["step"]}: castle star in {star[0]}' + (f' by {star[1]}' if star[1] else ''))
    elif (s['level'] is None or s['level'] in HUB) and (want := want_target(s)):
        want = VIA.get(want, want); skip = sorted(set(COURSE) - {want})
        if zp := zone_path(s['route'], want): env['OBRRT_ZONE_PATH'] = zp; log(f'step {s["step"]}: to {LV[want]} by {zp}')
    if skip: env['OBRRT_EXIT_SKIP'] = ','.join(str(x) for x in skip)
    if A.stall and s['level'] is not None and s['level'] not in HUB and not coins: env['OBRRT_STOP_STALL'] = str(int(A.stall))
    if final(s) and s['level'] in (BITS, ARENA[BITS]):
        # The last search: only the grand star's exit into the ending counts (search/src/tree.rs OBRRT_FINAL).
        env['OBRRT_FINAL'] = '1'
    if coins:
        # A coin leg: the 100-coin macro on (coins as work in progress), stopping on the star (it doesn't end the visit).
        env['OBRRT_TWOSTAR'] = '1'; env['OBRRT_STOP_PROGRESS'] = '1'; env['OBRRT_EXIT_NOSTOP'] = '1'
    if s.get('coin_got'):
        # The 100-coin star is had inside the course: any way out is good.
        env['OBRRT_ANY_EXIT'] = '1'
    cmd = ['nice', '-n', '5', A.bin, '../baserom.us.z64', out, str(s['step'] + 1), '600'] + ([s['route']] if s['route'] else [])
    with open(f'{D}/step-{s["step"]:04d}.log', 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    rows = list(csv.DictReader(open(out + '/nodes.csv'))) if os.path.exists(out + '/nodes.csv') else []
    try: used = json.load(open(out + '/summary.json'))['frames']
    except Exception: used = 0
    return out, rc, rows, used

def leave_by_pause(s, why):
    '''A star had inside the course (held, or the 100-coin star) and the route still in it: out by the pause menu
    (pause_exit). Updates the state and logs; False if the replay doesn't show it.'''
    route = f'{D}/route-{s["step"]:04d}.m64'
    r = pause_exit(s['route'], route)
    zone = s['level']
    if not r:
        log(f'step {s["step"]}: pause-menu exit from {LV.get(zone, zone)} not confirmed by replay'); return False
    lvl, stars, f = r
    got = [tuple(g) for g in s.get('held_got', [])]
    b = open(s['route'], 'rb').read(); n0 = len(b[0x400:] if b[:4] == b'M64\x1a' else b) // 4
    rec = dict(step=s['step'], zone=LV.get(zone, zone), stars=s['stars'], used=0.0, nodes=0, exits=1, to=LV.get(lvl, lvl), frame=f, took=f - n0, got=got, via='pause exit', note=f'pause menu ({why})')
    s.update(route=route, level=lvl, stars=stars, flags=s.get('held_flags', s['flags']), mult=1.0, coin_got=False, held=False, cp_n=0)
    s['skip'] = [l for l in s['skip'] if l != zone]
    s['history'].append(dict(rec, after=dict(stars=s['stars'], keys=s['keys'], caps=s.get('caps', 0)))); s['step'] += 1; save(s)
    log(f'step {rec["step"]}: {rec["zone"]} -> {rec["to"]} at frame {f} by the pause menu ({why}), stars {stars} NEW {got}')
    return True


def main():
    s = load()
    while s['step'] < A.steps and not s.get('done'):
        in_hub = s['level'] is None or s['level'] in HUB
        budget = (A.hub if in_hub else A.course) * s['mult']
        if s.get('coin_got') and s.get('held') and not in_hub and leave_by_pause(s, 'held'):
            continue
        star = castle_star(s) if in_hub and s['route'] else None
        moat = moat_step(s) if in_hub and s['route'] and not star and not final(s) else None
        # Both pillars have to be pounded: drained at 80.8 M in one test, not in 60 M or 150 M in two others.
        if moat is not None: budget = 3 * A.hub
        c = COURSE.get(s['level'] or 0, 0)
        coins = (A.coins > 0 and not in_hub and s['route'] and 1 <= c <= 15 and not s.get('coin_got') and s['level'] not in s.get('coin_skip', [])
                 and not flags(s['flags']).get(c, 0) & 0x40 and (s.get('coin') or s.get('cp_n', 0) == 0))
        if coins:
            s.setdefault('coin', None)
            if not s['coin']: s['coin'] = dict(level=s['level'], start=s['route'], n=0); log(f'step {s["step"]}: coin legs in {LV[s["level"]]} ({A.coins / 1e6:g} M frames each)')
            # After a leg gaining little, the retry has twice the frames: the coins left are further away
            # (BBH, sc1 steps 34-35: 52 -> 52 in 25 M, then nothing).
            # Each weak leg in a row doubles the next one (50, 100, 200 M with 25 M legs): the big jumps after a
            # stall all came from the longer leg (TTC 48 -> 73, THI 68 -> 78, BoB 81 -> 90, WDW 61 -> 71).
            budget = A.coins * 2 ** int(s['coin'].get('low') or 0)
        out, rc, rows, used = run_tree(s, budget, 0 if in_hub or coins else A.grace, star, coins, moat)
        s['frames'] += used
        if not rows:
            log(f'step {s["step"]}: the run made no tree (rc {rc}); stopping'); save(s); return 1
        if coins:
            root = rows[0]; zone = int(root['level']); cm = s['coin']
            # Retired nodes too (nodepath rebuilds any node from the tree's inputs; exits are in other levels):
            # a TTM leg (t1 step 219) had no live grounded state in 1196 nodes and the legs ended.
            # Not hurt: no knockback (ACT_FLAG_INVULNERABLE 0x20000) and 3 or more health segments (t1 step 398:
            # LLL at 93 coins from a hard knockback at 1 segment, every attempt dead at once, 7.6 hours stuck).
            inside = [r for r in rows[1:] if int(r['level']) in (zone, ARENA.get(zone, zone)) and int(r['action'], 16) & 0x1C0 in (0x000, 0x040)
                      and not int(r['action'], 16) & 0x20000 and int(r.get('health') or '0x880', 16) >= 0x300
                      # Not mid-slide (butt/stomach slide 0x40000, dive slide 0x80000): CCM's legs went on from Mario
                      # sliding down the slide at 53 coins; trees from there were a few thousand nodes, nothing gained.
                      and not int(r['action'], 16) & 0xC0000]
            # Not near a leg start that was a dead end (a tree of 1-2 nodes: t1 steps 777-778, SSL at 32 coins with
            # Mario sinking in quicksand at y -194; every attempt died, the legs restarted from it twice).
            dead = cm.get('dead', [])
            inside = [r for r in inside if not any(sum((float(r[k]) - q[i]) ** 2 for i, k in enumerate('xyz')) < 300 ** 2 for q in dead)]
            got = [r for r in inside if flags(r['star_flags']).get(c, 0) & 0x40]
            best = pick_feasible(out, inside, zone in SINGLE_AREA)
            # Into another area (one the course may not let Mario leave, like LLL's volcano): only with enough coins
            # there for 100, else the best state in the leg's own area. The user: LLL's legs went into the volcano at
            # 88 coins (leg 4) with about 15 left in it (103 at most); it got to 100 only after 2 weak legs.
            if best is not None and best['area'] != root['area'] and not area_has_coins(out, best, cm_with(cm, root, best)):
                same = [r for r in inside if r['area'] == root['area']]
                # Only when staying gains coins: DDD (t1 step 556) at 8 coins in area 2 was kept in area 1, which
                # gained 0 (its coins are in area 2; the estimate only sees what is loaded, and DDD's areas aren't
                # one-way).
                if same and max(int(r['coins']) for r in same) > int(root['coins']):
                    log(f'step {s["step"]}: not into area {best["area"]} at {best["coins"]} coins: too few coins there for 100')
                    best = max(same, key=lambda r: (int(r['coins']), -int(r['frame'])))
            # A one-way area (SSL's pyramid) gives up what is left in the area it was entered from: stay while the leg
            # gained there and many coins are left, so the outside is used up before the pyramid is entered.
            if best is not None and best['area'] != root['area'] and int(root['area']) not in reachable_areas(int(root['level']), int(best['area'])):
                lost = max(0, coin_inventory().get(COURSE.get(zone), {}).get(int(root['area']), 0) - int(cm_with(cm, root, best).get('got', {}).get(str(int(root['area'])), 0)))
                same = [r for r in inside if r['area'] == root['area']]
                if lost >= 10 and same and max(int(r['coins']) for r in same) - int(root['coins']) >= A.coin_gain:
                    log(f'step {s["step"]}: not into area {best["area"]} at {best["coins"]} coins: {lost} coins are left in area {root["area"]}, which has no way back')
                    best = max(same, key=lambda r: (int(r['coins']), -int(r['frame'])))
            gain = int(best['coins']) - int(root['coins']) if best else 0
            # Nothing gained here: the area may be used up (SSL's outside at 72). A state in another area that holds the
            # coins left for 100 (the inventory, not the loaded objects) is the next leg's start; before, the legs
            # went on from the state in the same area and stopped after 2 legs of 0-1 coins.
            if gain <= 0 and A.coin_area_move:
                moved = [r for r in inside if r['area'] != root['area'] and int(r['coins']) >= int(root['coins']) and area_has_coins(out, r, cm_with(cm, root, r))]
                if moved:
                    best = max(moved, key=lambda r: (int(r['coins']), -int(r['frame'])))
                    gain = int(best['coins']) - int(root['coins'])
                    log(f'step {s["step"]}: no coins gained in area {root["area"]}: on into area {best["area"]} at {best["coins"]} coins (the course holds {sum(coin_inventory().get(COURSE.get(zone), {}).values())})')
            if best is not None:
                # Booked to the area the leg began in (a leg that walks in at its end took its coins outside).
                g = cm.setdefault('got', {}); g[str(int(root['area']))] = int(g.get(str(int(root['area'])), 0)) + max(0, gain)
            # Low: fewer than --coin-gain coins, or than half of those still missing near the end (JRB, sc1
            # step 103: stopped at 98 of 104 after 95 -> 95 -> 98, 3 coins being 'low').
            low = gain < min(A.coin_gain, max(1, (100 - int(best['coins'] if best else root['coins'])) // 2))
            rec = dict(step=s['step'], zone=LV.get(zone, zone), stars=int(root['stars']), used=round(used / 1e6, 1), nodes=len(rows), coins=dict(leg=cm['n'] + 1, root=int(root['coins']), best=int(best['coins']) if best else None))
            cm['n'] += 1
            if got:
                pick = min(got, key=lambda r: int(r['frame']))
                route = f'{D}/coin-{s["step"]:04d}.m64'
                subprocess.run(['builds/tools/nodepath', out, pick['id'], route], check=True, stdout=subprocess.DEVNULL)
                s['route'] = route; s['coin'] = None; s['coin_got'] = True; s['cp_n'] = 0
                s['held_flags'] = pick['star_flags']; s['held_got'] = [[c, 7]]; s['held'] = False
                rec.update(got=[(c, 7)], frame=int(pick['frame']))
                log(f'step {s["step"]}: 100 COINS in {LV.get(zone, zone)} at frame {pick["frame"]} (leg {cm["n"]}, {rec["used"]} M frames); now out of the course')
            elif len(rows) <= 2 and cm.get('trail'):
                # A dead start: back to the leg before, and that spot is avoided from now on.
                cm.setdefault('dead', []).append([float(root[k]) for k in 'xyz'])
                s['route'] = cm['trail'].pop()
                log(f'step {s["step"]}: coin leg {cm["n"]} in {LV.get(zone, zone)}: a dead start at {[round(float(root[k])) for k in "xyz"]} ({len(rows)} nodes); back to the leg before')
                s['history'].append(dict(rec, after=dict(stars=s['stars'], keys=s['keys'], caps=s.get('caps', 0)))); s['step'] += 1; save(s)
                if not A.keep: shutil.rmtree(out, ignore_errors=True)
                continue
            elif int(cm.get('low') or 0) + low <= A.coin_retries and cm['n'] < A.coin_legs:
                # One leg gaining little gets another try from its best state (CCM, sc1 step 21: 0 coins in a leg).
                cm['low'] = int(cm.get('low') or 0) + 1 if low else 0
                if best is not None:
                    route = f'{D}/coin-{s["step"]:04d}.m64'
                    subprocess.run(['builds/tools/nodepath', out, best['id'], route], check=True, stdout=subprocess.DEVNULL)
                    cm.setdefault('trail', []).append(s['route'])
                    s['route'] = route
                log(f'step {s["step"]}: coin leg {cm["n"]} in {LV.get(zone, zone)}: {root["coins"]} -> ' + (f'{best["coins"]} coins at frame {best["frame"]}' if best else 'no state to go on from (again from the same one)') + f' ({rec["used"]} M frames, {len(rows)} nodes)')
            else:
                s.setdefault('coin_skip', []).append(cm['level']); s['route'] = cm['start']; s['coin'] = None
                log(f'step {s["step"]}: coin legs in {LV.get(zone, zone)} stop at {best["coins"] if best else "?"} coins (leg {cm["n"]} gained {gain}); the course from its entry')
            s['history'].append(dict(rec, after=dict(stars=s['stars'], keys=s['keys'], caps=s.get('caps', 0)))); s['step'] += 1; save(s)
            if not A.keep: shutil.rmtree(out, ignore_errors=True)
            continue
        if moat is not None:
            root = rows[0]
            got = [r for r in rows[1:] if int(r['level']) in HUB and int(r.get('save_flags') or '0', 16) & MOAT and int(r['action'], 16) & 0x1C0 != 0x100]
            rec = dict(step=s['step'], zone='Castle', stars=int(root['stars']), used=round(used / 1e6, 1), nodes=len(rows), exits=len(got))
            if got:
                pick = min(got, key=lambda r: int(r['frame']))
                route = f'{D}/route-{s["step"]:04d}.m64'
                subprocess.run(['builds/tools/nodepath', out, pick['id'], route], check=True, stdout=subprocess.DEVNULL)
                rec.update(to='Castle', frame=int(pick['frame']), took=int(pick['frame']) - int(root['frame']), got=[], via=pick['via'], note='moat drained')
                s.update(route=route, level=int(pick['level']), mult=1.0)
                log(f'step {s["step"]}: MOAT DRAINED at frame {pick["frame"]} (+{rec["took"]}); {rec["used"]} M frames, {len(rows)} nodes')
            else:
                s['moat_try'] = [s['stars'], s['keys']]
                log(f'step {s["step"]}: moat not drained in {rec["used"]} M frames ({len(rows)} nodes); on with the courses')
            s['history'].append(dict(rec, after=dict(stars=s['stars'], keys=s['keys'], caps=s.get('caps', 0)))); s['step'] += 1; save(s)
            if not A.keep and got: shutil.rmtree(out, ignore_errors=True)
            continue
        if star:
            root = rows[0]
            # In control (not in the cutscene action group: the star's dance, its save prompt), so the next tree boots.
            got = [r for r in rows[1:] if int(r['level']) in HUB and r['retired'] != '1' and int(r['stars']) > int(root['stars'])
                   and int(r['action'], 16) & 0x1C0 != 0x100]
            rec = dict(step=s['step'], zone=LV.get(int(root['level']), root['level']), stars=int(root['stars']), used=round(used / 1e6, 1), nodes=len(rows), exits=len(got))
            if got:
                pick = min(got, key=lambda r: int(r['frame']))
                route = f'{D}/route-{s["step"]:04d}.m64'
                subprocess.run(['builds/tools/nodepath', out, pick['id'], route], check=True, stdout=subprocess.DEVNULL)
                rec.update(to='Castle', frame=int(pick['frame']), took=int(pick['frame']) - int(root['frame']), got=['castle'], via=pick['via'])
                s.update(route=route, level=int(pick['level']), stars=int(pick['stars']), keys=int(pick['keys']), flags=pick['star_flags'], mult=1.0)
                log(f'step {s["step"]}: castle star at frame {rec["frame"]} (+{rec["took"]}), stars {s["stars"]} NEW castle; {rec["used"]} M frames, {len(rows)} nodes')
            else:
                s['castle_try'] = [s['stars'], s['keys']]
                log(f'step {s["step"]}: no castle star in {rec["used"]} M frames ({len(rows)} nodes); on with the courses')
            s['history'].append(dict(rec, after=dict(stars=s['stars'], keys=s['keys'], caps=s.get('caps', 0)))); s['step'] += 1; save(s)
            if not A.keep and got: shutil.rmtree(out, ignore_errors=True)
            continue
        root = rows[0]; zone = int(root['level']); inside = HUB if zone in HUB else {zone, ARENA.get(zone, zone)}
        if final(s) and zone in (BITS, ARENA[BITS]):
            ending = [r for r in rows[1:] if int(r['level']) == 16]
            if ending:
                # The grand star: the route to it, then the ending played out by idle frames and checked by
                # replay (tools/seg120/credits.py), and the campaign is over.
                pick = min(ending, key=lambda r: int(r['frame']))
                route = f'{D}/route-{s["step"]:04d}.m64'
                subprocess.run(['builds/tools/nodepath', out, pick['id'], route], check=True, stdout=subprocess.DEVNULL)
                cr = subprocess.run(['python3', 'tools/seg120/credits.py', route, f'{D}/final.m64'], capture_output=True, text=True)
                rec = dict(step=s['step'], zone=LV.get(zone, zone), stars=int(root['stars']), used=round(used / 1e6, 1), nodes=len(rows), to='Ending',
                           frame=int(pick['frame']), took=int(pick['frame']) - int(root['frame']), got=['grand star'], via=pick['via'])
                s.update(route=route, level=16, done=True)
                s['history'].append(dict(rec, after=dict(stars=s['stars'], keys=s['keys'], caps=s.get('caps', 0)))); s['step'] += 1; save(s)
                log(f'step {rec["step"]}: GRAND STAR at frame {pick["frame"]} (+{rec["took"]}); {rec["used"]} M frames, {len(rows)} nodes')
                log('credits: ' + (cr.stdout.strip() or cr.stderr.strip()) + (f'; THE END: {D}/final.m64' if cr.returncode == 0 else '; NOT CHECKED'))
                return 0 if cr.returncode == 0 else 3
        # Exits: retired at once by the lock, in a level outside the zone.
        ex = [r for r in rows[1:] if int(r['level']) not in inside and r['retired'] == '1']
        pick = None
        want = want_target(s) if zone in HUB else None
        via = VIA.get(want, want)
        # An entry into a cap course with its switch unpressed, from a course (see VIA).
        cap_in = lambda r: bool(CAP.get(int(r['level']), 0) & ~int(r.get('caps') or 0)) and int(r['level']) not in s['skip']
        need = CAP.get(zone, 0) & ~s.get('caps', 0)
        if zone in HUB:
            ex = [r for r in ex if (int(r['level']) == via if want else int(r['level']) not in no_go(s))]
            if ex: pick = min(ex, key=lambda r: int(r['frame']))
        else:
            # Against the campaign's flags, not the root's: a 100-coin star had by the coin legs counts here.
            good = [(r, new_stars(s['flags'], r['star_flags']), int(r['keys']) - int(root['keys'])) for r in ex]
            # A cap-switch course with its switch unpressed: only an exit with it pressed (see CAP).
            if need: good = [(r, st, k + 1) for r, st, k in good if int(r.get('caps') or 0) & need]
            else: good = [(r, st, k + cap_in(r)) for r, st, k in good if st or k > 0 or cap_in(r)]
            if good: pick = min(good, key=lambda g: (int(g[0]['frame']) - int(root['frame'])) / (len(g[1]) + g[2]))[0]
        rec = dict(step=s['step'], zone=LV.get(zone, zone), stars=int(root['stars']), used=round(used / 1e6, 1), nodes=len(rows), exits=len(ex))
        # Every useful exit is kept as inputs from power-on (the segment library's raw material), not only
        # the one the route takes: the fastest of each (level entered, stars gained).
        alts = {}
        for r in ex:
            got_r = tuple(new_stars(root['star_flags'], r['star_flags'])); dk = int(r['keys']) - int(root['keys'])
            if zone not in HUB and not got_r and dk <= 0: continue
            k = (int(r['level']), r['act'] if zone in HUB else '', got_r, dk)
            if k not in alts or int(r['frame']) < int(alts[k]['frame']): alts[k] = r
        os.makedirs(f'{D}/exits', exist_ok=True)
        for (lvl, act, got_r, dk), r in sorted(alts.items(), key=lambda kv: int(kv[1]['frame']))[:16]:
            f = f'{D}/exits/s{s["step"]:04d}-n{r["id"]}.m64'
            subprocess.run(['builds/tools/nodepath', out, r['id'], f], check=True, stdout=subprocess.DEVNULL)
            open(f'{D}/exits/index.jsonl', 'a').write(json.dumps(dict(step=s['step'], file=f, zone=zone, root_frame=int(root['frame']), frame=int(r['frame']),
                to=lvl, act=act, got=list(got_r), keys=dk, stars=int(r['stars']), flags=r['star_flags'], from_flags=root['star_flags'])) + '\n')
        if pick is not None:
            route = f'{D}/route-{s["step"]:04d}.m64'
            subprocess.run(['builds/tools/nodepath', out, pick['id'], route], check=True, stdout=subprocess.DEVNULL)
            got = new_stars(s['flags'] if zone not in HUB else root['star_flags'], pick['star_flags']); s['coin_got'] = False
            rec.update(to=LV.get(int(pick['level']), pick['level']), frame=int(pick['frame']), took=int(pick['frame']) - int(root['frame']), got=got, via=pick['via'])
            if zone in HUB: s['hub_route'] = s['route']; s['hub_level'] = zone
            s['cp_n'] = 0
            # A star opens the next act of its own course, so that course is tried again; the others stay
            # skipped until the castle has nothing else (then the list is cleared and budgets doubled).
            caps = int(pick.get('caps') or s.get('caps', 0)); cap_new = caps & ~s.get('caps', 0)
            if cap_new: rec['caps'] = caps
            if got or int(pick['keys']) > s['keys'] or cap_new: s['skip'] = [l for l in s['skip'] if l not in (zone, s['level'])]; s['mult'] = 1.0; s['open_fail'] = []
            s.update(route=route, level=int(pick['level']), stars=int(pick['stars']), keys=int(pick['keys']), flags=pick['star_flags'], caps=caps)
            # A course given up after checkpoints is resumed from its last one, while nothing was gained since.
            kept = s.get('cps', {}).get(str(s['level']))
            if zone in HUB and kept and (kept['flags'], kept['keys']) == (s['flags'], s['keys']) and os.path.exists(kept['route']):
                s['route'] = kept['route']; rec['resumed'] = kept['route']
            log(f'step {s["step"]}: {rec["zone"]} -> {rec["to"]} at frame {rec["frame"]} (+{rec["took"]}), stars {s["stars"]}' + (f' NEW {got}' if got else '') + (' CAP SWITCH' if cap_new else '') + f'; {rec["used"]} M frames, {len(rows)} nodes, {len(ex)} exits')
        elif zone in HUB and want:
            # The cap course (or the open course, want_open) wasn't reached: given up for now, like any course.
            s['skip'] = sorted(set(s['skip']) | {want})
            if want not in CAP: s.setdefault('open_fail', []).append(want)
            log(f'step {s["step"]}: castle did not reach {LV[want]} ({"cap switch" if want in CAP else "most stars missing"}) in {rec["used"]} M frames; skipping it')
        elif zone in HUB:
            # Nothing left that the castle reaches and we haven't given up on: try harder.
            s['skip'] = []; s['mult'] *= 2
            log(f'step {s["step"]}: castle found no new course in {rec["used"]} M frames; budgets x{s["mult"]:g}')
            if s['mult'] > 8:
                log('giving up: budgets above x8'); s['history'].append(dict(rec, after=dict(stars=s['stars'], keys=s['keys'], caps=s.get('caps', 0)))); save(s); return 2
        elif zone not in HUB and (held := [r for r in rows[1:] if int(r['level']) in inside and int(r['action'], 16) & 0x1C0 in (0x000, 0x040)
                                             and new_stars(s['flags'], r['star_flags'])]):
            # A star had without leaving the course (a Bowser course's red-coin star, like the 100-coin
            # star, doesn't end the visit): go on from the earliest state in control with it, and the next
            # tree only has to leave (OBRRT_ANY_EXIT; its exit counts the star). sc1 step 132: BitFS's red
            # coin star at 85 frames from the root, then no exit in 150 M, and the course was skipped.
            pick_h = min(held, key=lambda r: int(r['frame']))
            route = f'{D}/held-{s["step"]:04d}.m64'
            subprocess.run(['builds/tools/nodepath', out, pick_h['id'], route], check=True, stdout=subprocess.DEVNULL)
            s['route'] = route; s['coin_got'] = True; s['cp_n'] = 0
            s['held_flags'] = pick_h['star_flags']; s['held_got'] = [list(g) for g in new_stars(s['flags'], pick_h['star_flags'])]; s['held'] = True
            rec.update(held=new_stars(s['flags'], pick_h['star_flags']), frame=int(pick_h['frame']))
            log(f'step {s["step"]}: {rec["zone"]} star {rec["held"]} had at frame {pick_h["frame"]} without leaving ({rec["used"]} M frames); next: out of the course')
        elif zone not in HUB and s.get('cp_n', 0) < A.checkpoints and (cp := (cannon_cp(out, root) or stage_cp(out, root, zone, dead_ends(s, zone), s.get('cp_seen', {}).get(str(zone), {}).get('stages', [])) or checkpoint(out, root, zone, s.get('cp_n', 0), s.get('cp_goal') if s.get('cp_n', 0) else None,
                                                                                      dead_ends(s, zone), s['keys'], final(s)))) is not None:
            # No exit, but further along a goal: a fresh tree goes on from there.
            route = f'{D}/cp-{s["step"]:04d}.m64'
            subprocess.run(['builds/tools/nodepath', out, str(cp['node']), route], check=True, stdout=subprocess.DEVNULL)
            s['cp_n'] = s.get('cp_n', 0) + 1; s['route'] = route
            seen = s.setdefault('cp_seen', {}).get(str(zone))
            if not seen or not same_course(seen['flags'], s['flags'], zone): seen = s['cp_seen'][str(zone)] = dict(flags=s['flags'], at=[])
            seen['at'].append([cp['level'], [round(x) for x in cp['pos']]])
            if cp['how'] == 'stage': seen.setdefault('stages', []).append(cp['kind'])
            s['cp_goal'] = [cp['key'], cp['how']] if cp.get('key') and cp['how'] != 'moved' else None
            s.setdefault('cps', {})[str(s['level'])] = dict(route=route, flags=s['flags'], keys=s['keys'])
            rec.update(cp=dict(node=cp['node'], goal=cp['kind'], how=cp['how'], frame=cp['frame'], level=cp['level'], eta=cp['eta'], eta0=cp['eta0'], wip=cp['wip']))
            e0 = 'none' if cp['eta0'] is None else f'{cp["eta0"]:.0f}'
            log(f'step {s["step"]}: {rec["zone"]} no exit in {rec["used"]} M frames ({len(rows)} nodes); checkpoint {s["cp_n"]} at node {cp["node"]}, frame {cp["frame"]} (+{cp["frame"] - int(root["frame"])}), '
                f'{LV.get(cp["level"], cp["level"])} {[round(x) for x in cp["pos"]]}: {cp["kind"]} ({cp["how"]}) ETA {e0} -> {cp["eta"]:.0f}, wip {cp["wip0"]} -> {cp["wip"]}')
        else:
            if s.get('coin_got') and leave_by_pause(s, 'no exit found'):
                continue
            # The course gave nothing: back to the castle state before it, and skip it for now.
            zone = s['level'] if s['level'] not in HUB else zone; s['cp_n'] = 0; s['coin_got'] = False
            s['skip'] = sorted(set(s['skip']) | {zone} | {c for c, v in VIA.items() if v == zone}); s['route'] = s['hub_route']; s['level'] = s.get('hub_level', 6)
            log(f'step {s["step"]}: {rec["zone"]} gave no star in {rec["used"]} M frames ({len(rows)} nodes, {len(ex)} exits); skipping it, back to the castle')
        s['history'].append(dict(rec, after=dict(stars=s['stars'], keys=s['keys'], caps=s.get('caps', 0)))); s['step'] += 1; save(s)
        if not A.keep and pick is not None: shutil.rmtree(out, ignore_errors=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
