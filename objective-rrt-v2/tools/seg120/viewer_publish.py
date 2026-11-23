#!/usr/bin/env python3
"""Put a campaign's current route on sm64-viewer's start page: one entry in index.json, updated in place.

    tools/seg120/viewer_publish.py [NAME] [--viewer <viewer>/frontend/public/replays] [--dry-run]

NAME (default t1) is the campaign in logs/campaign/NAME. Reads its state.json and replays its current route with
builds/tools/tasplay (nice 19, one at a time) to get the frames of every level change, star and key. Then:
- copies the route to <viewer>/obrrt2/campaign-NAME.m64 (the same name every time, replaced atomically);
- writes the regions: one band per level visit, named by the level, its note the campaign step that made it and
  the frames it searched; and, as viewer_routes.py does, a "Star N" band from each star (and "Key N" from a key);
- inserts or replaces its own entry in index.json (the one whose `file` is that movie): the text gives the stars,
  keys, route length and frames searched, and a results table plays the whole run and the EXTRAS below (other
  movies of the campaign, each opened near its interesting part).
Safe to rerun while the campaign runs: nothing else in index.json or the replay folder is touched, and nothing
under logs/ is written. The same state gives the same output.
"""
import argparse, csv, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('name', nargs='?', default='t1')
ap.add_argument('--viewer', default=str(HERE.parents[1] / 'sm64-viewer/frontend/public/replays'))
ap.add_argument('--dry-run', action='store_true', help='print the entry, write nothing')
A = ap.parse_args()
VIEWER = Path(A.viewer); OUT = VIEWER / 'obrrt2'
DATE = '2026-11-10'
TITLE = '120-star campaign (self-routed, no TAS inputs)'
MAIN = f'obrrt2/campaign-{A.name}.m64'

LV = {9:'BoB',24:'WF',12:'JRB',5:'CCM',4:'BBH',7:'HMC',22:'LLL',8:'SSL',23:'DDD',10:'SL',11:'WDW',36:'TTM',13:'THI',14:'TTC',15:'RR',27:'PSS',6:'Castle',16:'Grounds',26:'Courtyard',29:'TotWC',28:'CotMC',18:'VCutM',31:'WMotR',20:'SA',17:'BitDW',19:'BitFS',21:'BitS',30:'Bowser1',33:'Bowser2',34:'Bowser3'}
NUM = {v: k for k, v in LV.items()}
FULL = {1:'Menus', 16:'Castle grounds', 6:'Castle', 26:'Courtyard', 9:'Bob-omb Battlefield', 24:"Whomp's Fortress", 12:'Jolly Roger Bay', 5:'Cool, Cool Mountain', 4:"Big Boo's Haunt",
        7:'Hazy Maze Cave', 22:'Lethal Lava Land', 8:'Shifting Sand Land', 23:'Dire, Dire Docks', 10:"Snowman's Land", 11:'Wet-Dry World', 36:'Tall, Tall Mountain', 13:'Tiny-Huge Island',
        14:'Tick Tock Clock', 15:'Rainbow Ride', 27:"Princess's Secret Slide", 29:'Tower of the Wing Cap', 28:'Cavern of the Metal Cap', 18:'Vanish Cap under the Moat',
        31:'Wing Mario over the Rainbow', 20:'Secret Aquarium', 17:'Bowser in the Dark World', 19:'Bowser in the Fire Sea', 21:'Bowser in the Sky', 30:'Bowser 1', 33:'Bowser 2', 34:'Bowser 3'}
AREA = {(8, 2): 'Inside the pyramid'}        # a band of its own inside a level visit
HUB, HUBZ = {1, 6, 16, 26}, {'Castle', 'Grounds', 'Courtyard'}
COURSE = {9:1,24:2,12:3,5:4,4:5,7:6,22:7,8:8,23:9,10:10,11:11,36:12,13:13,14:14,15:15,17:16,19:17,21:18,27:19,28:20,29:21,18:22,31:23,20:24}
SHORT = {c: LV[l] for l, c in COURSE.items()}
STAR = {1: ["Big Bob-omb on the Summit", "Footrace with Koopa the Quick", "Shoot to the Island in the Sky", "Find the 8 Red Coins", "Mario Wings to the Sky", "Behind Chain Chomp's Gate"],
        2: ["Chip Off Whomp's Block", "To the Top of the Fortress", "Shoot into the Wild Blue", "Red Coins on the Floating Isle", "Fall onto the Caged Island", "Blast Away the Wall"],
        3: ["Plunder in the Sunken Ship", "Can the Eel Come Out to Play?", "Treasure of the Ocean Cave", "Red Coins on the Ship Afloat", "Blast to the Stone Pillar", "Through the Jet Stream"],
        4: ["Slip Slidin' Away", "Li'l Penguin Lost", "Big Penguin Race", "Frosty Slide for 8 Red Coins", "Snowman's Lost His Head", "Wall Kicks Will Work"],
        5: ["Go on a Ghost Hunt", "Ride Big Boo's Merry-Go-Round", "Secret of the Haunted Books", "Seek the 8 Red Coins", "Big Boo's Balcony", "Eye to Eye in the Secret Room"],
        6: ["Swimming Beast in the Cavern", "Elevate for 8 Red Coins", "Metal-Head Mario Can Move!", "Navigating the Toxic Maze", "A-Maze-ing Emergency Exit", "Watch for Rolling Rocks"],
        7: ["Boil the Big Bully", "Bully the Bullies", "8-Coin Puzzle with 15 Pieces", "Red-Hot Log Rolling", "Hot-Foot-It into the Volcano", "Elevator Tour in the Volcano"],
        8: ["In the Talons of the Big Bird", "Shining Atop the Pyramid", "Inside the Ancient Pyramid", "Stand Tall on the Four Pillars", "Free Flying for 8 Red Coins", "Pyramid Puzzle"],
        9: ["Board Bowser's Sub", "Chests in the Current", "Pole-Jumping for Red Coins", "Through the Jet Stream", "The Manta Ray's Reward", "Collect the Caps..."],
        10: ["Snowman's Big Head", "Chill with the Bully", "In the Deep Freeze", "Whirl from the Freezing Pond", "Shell Shreddin' for Red Coins", "Into the Igloo"],
        11: ["Shocking Arrow Lifts!", "Top o' the Town", "Secrets in the Shallows & Sky", "Express Elevator--Hurry Up!", "Go to Town for Red Coins", "Quick Race Through Downtown!"],
        12: ["Scale the Mountain", "Mystery of the Monkey Cage", "Scary 'Shrooms, Red Coins", "Mysterious Mountainside", "Breathtaking View from Bridge", "Blast to the Lonely Mushroom"],
        13: ["Pluck the Piranha Flower", "The Tip Top of the Huge Island", "Rematch with Koopa the Quick", "Five Itty Bitty Secrets", "Wiggler's Red Coins", "Make Wiggler Squirm"],
        14: ["Roll into the Cage", "The Pit and the Pendulums", "Get a Hand", "Stomp on the Thwomp", "Timed Jumps on Moving Bars", "Stop Time for Red Coins"],
        15: ["Cruiser Crossing the Rainbow", "The Big House in the Sky", "Coins Amassed in a Maze", "Swingin' in the Breeze", "Tricky Triangles!", "Somewhere over the Rainbow"]}
CASTLE_DIALOG = {84: 'MIPS the rabbit', 162: 'MIPS the rabbit', 82: 'Toad', 76: 'Toad', 83: 'Toad'}
GREY, GOLD, PALETTE = '#8b93a7', '#f5a623', ['#4c8bf5', '#2bb673', '#e0568a', '#9b6bf2', '#35b8c9']
LEAD = 90       # a row opens this many frames before its level change

# The Bowser 1 section of t1 (step 147, put in by hand) joins three trees; the joins, from RUNS.md and t1/log.txt.
BY_HAND = [(51698, 'BitDW: second tree', "Checkpoint: the first tree's furthest state, after 94.5 M frames with no exit. A second tree starts here; it reaches the arena and hits Bowser in 21.0 M frames."),
           (53610, 'Bowser 1: third tree', 'Checkpoint: Bowser is hit. A third tree starts here and takes the key in 4.2 M frames.')]
# Other movies shown as table rows. `hist`: the history records its route is made of, as (campaign, last step).
# `open`: where the row starts playing, ('level', n) = the last entry into level n, ('area', level, area) likewise.
# Numbers in the texts that are not computed here are from RUNS.md and logs/campaign/t1/log.txt.
EXTRAS = [
    dict(src='logs/campaign/t1/route-0147-key1.m64', dst='campaign-t1-key1.m64', name='Route to Bowser key 1', hist=[('t1', 147)], open=('level', 17), search='94.5 + 21.0 + 4.2 M',
         text="Opens at the fall into Bowser in the Dark World (frame {enter}). The course, the Bowser 1 fight, the key at frame {key} and back in the castle at frame {total}. "
              "Joined by hand from three of our own trees at their furthest states: the first got about half way up the course in 94.5 M frames, the second reached the arena and hit Bowser, the third took the key."),
    dict(src='logs/campaign/jrbcp/route-0000.m64', dst='campaign-jrbcp-red-coins.m64', name='JRB red coins (side campaign jrbcp)', hist=[('t1', 147), ('jrbcp', 0)], open=('level', 12), used=('jrbcp', 0),
         text="Opens at the Jolly Roger Bay entry (frame {enter}). Red Coins on the Ship Afloat: the 8 red coins and the star at frame {star}, back in the castle at frame {total}. "
              "One tree with the experience roadmap. Earlier trees without it found no star here (the last two: 110.2 M and 174.3 M frames). One run, so the roadmap's share is not measured. "
              "Not in the main route: it branches off at 19 stars."),
    dict(src='logs/campaign/coin1/route-0006.m64', dst='campaign-coin1-wf-100-coins.m64', name='WF 100 coins by coin legs (test campaign coin1)',
         hist=[('t1', 178), ('sc1', 0), ('coin1', 6)], open=('level', 24), search='120.1 + 32.0 M',
         text="Opens at the Whomp's Fortress entry (frame {enter}). The 100-coin star by five short trees ('coin legs', 25 M frames each, each going on from the state "
              "with the most coins: 0, 29, 49, 75, 92, then 100), then a tree that leaves the course and takes the red-coin star on the way (frame {star}); back in the castle "
              "at frame {total}. A single tree had 81 coins at 300 M frames and no star. Not in the main route: it branches off at 31 stars."),
    dict(src='logs/campaign/sc1/route-0074.m64', dst='campaign-sc1-bob-100-coins-wing.m64', name='BoB 100 coins with the wing cap (scout sc1)',
         hist=[('t1', 178), ('sc1', 74)], open=('level', 9), search='~342 M',
         text="Opens at the Bob-omb Battlefield entry (frame {enter}). The 100-coin star by eleven coin legs, 43 of the coins from the sky rings: the wing cap from its box, "
              "a cannon shot and flight through the rings (new objective TwoStar::WingRing). Then the wing cap star too, out at frame {total}. "
              "Before the wing cap was part of the coin targets, BoB's coin legs stopped at 41. Not in the main route (scout sc1)."),
    dict(src='logs/campaign/t1/cp-0153.m64', dst='campaign-t1-ssl-pyramid.m64', name='Failure: SSL, Inside the Ancient Pyramid', hist=[('t1', 152)], open=('area', 8, 2), used=('t1', 153),
         tail='Step 153: no exit in 123.3 M frames. Inside the pyramid the tree got no higher than y 1244; the star is at y 5050.',
         text="Opens at the pyramid entry (frame {enter}). No star: inside the pyramid the tree got no higher than y 1244, and the star is at y 5050. The movie ends inside the pyramid, at the bottom."),
]


def clock(f): return f'{f // 1800}:{f % 1800 / 30:04.1f}'


def load_state(name):
    """state.json of a campaign; the campaign replaces it as it goes, so a failed read is retried."""
    for _ in range(20):
        try: return json.load(open(HERE / f'logs/campaign/{name}/state.json'))
        except (OSError, ValueError): time.sleep(0.25)
    sys.exit(f'cannot read logs/campaign/{name}/state.json')


def running(name):
    """Whether a campaign.py NAME process is alive (looks at /proc only)."""
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try: a = p.read_bytes().split(b'\0')
        except OSError: continue
        if any(x.endswith(b'campaign.py') for x in a) and name.encode() in a: return True
    return False


class Scan:
    """What tasplay sees in a movie: per-frame level, stars and keys; level visits; areas of AREA; pickups."""
    def __init__(self, m64, tmp):
        out = Path(tmp) / 'play.csv'
        subprocess.run(['nice', '-n', '19', str(HERE / 'builds/tools/tasplay'), str(HERE.parent / 'baserom.us.z64'), str(m64), str(out)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.level, self.stars, self.keys = [0], [0], [0]          # index = frame (frames count from 1)
        self.visits, self.areas, self.picks, self.control = [], [], [], None
        cur = area = None; since = 0; said = (None, -10**9)
        with open(out) as fh:
            rd = csv.reader(fh); h = next(rd); c = {k: h.index(k) for k in ('frame', 'level', 'area', 'course', 'stars', 'keys', 'dialog', 'controllable', 'save_flags')}
            self.save_flags = 0
            for row in rd:
                f, l, a, s, k, d = (int(row[c[x]]) for x in ('frame', 'level', 'area', 'stars', 'keys', 'dialog'))
                assert f == len(self.level), f'frame {f} out of order in {m64}'
                if l and l != cur: cur, area, since = l, a, f; self.visits.append((f, l))
                elif l and a != area:
                    # The first frames of a visit report a stale area; a real change comes later.
                    if f - since > 60 and ((l, a) in AREA or (l, area) in AREA): self.areas.append((f, l, a))
                    area = a
                if d >= 0: said = (d, f)
                if self.control is None and row[c['controllable']] == '1': self.control = f
                for n in range(self.stars[-1] + 1, s + 1):
                    self.picks.append(dict(kind='star', n=n, frame=f, level=l, course=int(row[c['course']]), who=CASTLE_DIALOG.get(said[0]) if f - said[1] < 300 else None))
                for n in range(self.keys[-1] + 1, k + 1): self.picks.append(dict(kind='key', n=n, frame=f, level=l))
                self.level.append(l); self.stars.append(s); self.keys.append(k); self.save_flags = int(row[c['save_flags']], 16)
        out.unlink()
        self.total = len(self.level) - 1

    def last_entry(self, where):
        if where[0] == 'level': return max((f for f, l in self.visits if l == where[1]), default=None)
        return max((f for f, l, a in self.areas if (l, a) == where[1:]), default=None)


def chain(hist, sc):
    """The history records the movie is made of, oldest first, as (index in hist, frame it started from, record):
    the last record whose exit the movie reaches, then back through each record's root (frame - took)."""
    i = next((i for i in range(len(hist) - 1, -1, -1) if 'to' in hist[i] and hist[i]['frame'] <= sc.total and sc.level[hist[i]['frame']] == NUM.get(hist[i]['to'])), None)
    out = []
    while i is not None:
        r = hist[i]
        if 'took' in r:
            p = r['frame'] - r['took']; j = next((j for j in range(i - 1, -1, -1) if 'to' in hist[j] and hist[j]['frame'] == p), None)
        else:       # put in by hand: it follows the last entry into its zone at the same star count
            z = r['zone'].split('+')[0]
            j = next((j for j in range(i - 1, -1, -1) if hist[j].get('to') == z and hist[j]['frame'] < r['frame'] and hist[j]['stars'] == r['stars']), None)
            p = hist[j]['frame'] if j is not None else 0
        out.append((i, p, r)); i = j
    # A record whose star count the movie doesn't have at that frame isn't this movie's: drop it.
    return [(i, p, r) for i, p, r in reversed(out) if sc.stars[min(max(p, 1), sc.total)] == r['stars']]


def step_note(hist, i, r):
    t = f"Step {r['step']}: "
    if r['zone'] in HUBZ: t += f"castle to {FULL.get(NUM.get(r['to']), r['to'])}, {r['used']} M frames searched."
    else:
        t += f"{r['used']} M frames searched."
        z = r['zone'].split('+')[0]
        lost = [h['used'] for h in hist[:i] if h['zone'] == z and 'to' not in h and 'cp' not in h and h['stars'] == r['stars']]
        if lost: t += f" Before it, {len(lost)} search{'es' if len(lost) > 1 else ''} here at {r['stars']} stars ended with no new star or key ({sum(lost):.1f} M frames)."
    if r.get('note'): t += ' ' + r['note'][0].upper() + r['note'][1:].rstrip('.') + '.'
    return t


def regions(sc, hist, tail=''):
    """One band per level visit (and per AREA inside it), plus a band from each star and key."""
    ch = chain(hist, sc)
    def step_at(f): return next(((i, r) for i, p, r in ch if p < f <= r['frame']), None)
    color, out = {}, []
    def col(l): return GREY if l in HUB else color.setdefault(l, PALETTE[len(color) % len(PALETTE)])
    ends = [f for f, _ in sc.visits[1:]] + [None]
    for (f, l), end in zip(sc.visits, ends):
        if end is None: note = tail
        elif (s := step_at(end)) is not None: note = step_note(hist, *s)
        else: note = "The search's own boot inputs." if ch and end <= ch[0][1] else ''
        out.append(dict({'from': f}, label=FULL.get(l, f'Level {l}'), note=note, color=col(l)))
        out += [dict({'from': g}, label=AREA.get((l, a), FULL.get(l, f'Level {l}')), note=note, color=col(l)) for g, m, a in sc.areas if m == l and f < g < (end or sc.total + 1)]
    if any(r['step'] == 147 and r.get('key') for _, _, r in ch):
        out += [dict({'from': f}, label=label, note=note, color=col(sc.level[f])) for f, label, note in BY_HAND if f <= sc.total]
    for p in sc.picks:
        at = f"Frame {p['frame']}, {clock(p['frame'])}"
        if p['kind'] == 'key': out.append(dict({'from': p['frame']}, label=f"Key {p['n']}", note=f"Bowser key {p['n']}. {at}", color=GOLD)); continue
        s = step_at(p['frame']); got = s[1].get('got', []) if s else []
        if p['level'] in HUB: label, name = f"Star {p['n']}: castle", (p['who'] + "'s star" if p['who'] else 'Castle secret star')
        elif len(got) == 1 and got[0][0] == p['course']:
            c, k = got[0]; label = f"Star {p['n']}: {SHORT.get(c, c)}" + (f' {k}' if c in STAR else '')
            name = (STAR[c][k - 1] if k <= 6 else '100 coins') if c in STAR else FULL.get(p['level'], '')
        else: label, name = f"Star {p['n']}: {SHORT.get(p['course'], p['course'])}", FULL.get(p['level'], '')
        out.append(dict({'from': p['frame']}, label=label, note=f"{name}{'' if name[-1] in '?!.' else '.'} {at}" if name else at, color=GOLD))
    out.sort(key=lambda r: (r['from'], r['color'] == GOLD))
    return out


def star_list(s):
    """'BoB 1, 3, 6; WF ...' from the state's flags, and the castle stars they don't hold."""
    parts, n = [], 0
    for p in s['flags'].split():
        c, m = p.split(':'); c, m = int(c), int(m, 16) & 0x7f; got = [i + 1 for i in range(7) if m >> i & 1]; n += len(got)
        if got: parts.append(SHORT.get(c, f'course {c}') + (' ' + ', '.join(map(str, got)) if c in STAR or len(got) > 1 else ''))
    if s['stars'] > n: parts.append(f"{s['stars'] - n} in the castle")
    return '; '.join(parts)


# The 120 stars for the start page's star board: the main courses (6 act stars and the 100-coin star each), the
# secret stars, the castle's five, by course number (save file order), and the cap switches.
SECRET = [(16, 'Bowser in the Dark World', ['8 Red Coins']), (17, 'Bowser in the Fire Sea', ['8 Red Coins']), (18, 'Bowser in the Sky', ['8 Red Coins']),
          (19, "Princess's Secret Slide", ['Finish the slide', 'Under 21 seconds']), (20, 'Cavern of the Metal Cap', ['8 Red Coins']),
          (21, 'Tower of the Wing Cap', ['8 Red Coins']), (22, 'Vanish Cap under the Moat', ['8 Red Coins']),
          (23, 'Wing Mario over the Rainbow', ['8 Red Coins']), (24, 'Secret Aquarium', ['8 Red Coins'])]
CASTLE = [(24, "Toad (12 stars, basement)"), (25, 'Toad (25 stars, upstairs)'), (26, 'Toad (50 stars, top floor)'),
          (27, 'MIPS the rabbit (15 stars)'), (28, 'MIPS the rabbit (50 stars)')]
SWITCHES = [(1, 'Wing cap switch'), (2, 'Metal cap switch'), (3, 'Vanish cap switch')]
LONG = {c: FULL[l] for l, c in COURSE.items()}


def board(s, sc, hist):
    """Which of the 120 stars the route has and which are left: from the campaign's flags (course stars) and the
    replay's save flags at its last frame (castle stars, switches). A star the route has gets the campaign step and
    the route frame it was collected at, where the history says (a pickup in that step's part of the route)."""
    have = flags_of(s['flags'])
    # A record that ended with stars is part of this route when the replay has its star count at its exit
    # frame; its stars are the pickups in its window. (The history chain breaks at hand joins and checkpoints.)
    when = {}; prev = None
    for r in hist:
        got = [tuple(g) for g in r.get('got', []) if isinstance(g, (list, tuple)) and tuple(g) not in when]
        # A coin-leg record (the 100-coin star, no `took`): its node is in control after the star's dance. A
        # record joined by hand (no `took`; t1 step 607, the Koopa race): its window starts at the record before.
        f1, took = r.get('frame'), r.get('took', 400 if 'coins' in r else None)
        if took is None and f1 is not None and prev is not None and f1 > prev: took = f1 - prev
        if f1 is not None: prev = f1
        if not got or f1 is None or took is None or f1 > sc.total: continue
        frames = [q['frame'] for q in sc.picks if q['kind'] == 'star' and f1 - took < q['frame'] <= f1 and not q.get('who')]
        # Part of this route: the stars picked up in the window are the record's (a star had inside the
        # course before, like the 100-coin star, isn't picked up again at its exit).
        if len(frames) != len(got) or sc.stars[f1] < len(frames): continue
        for g, f in zip(got, frames): when[g] = dict(step=r['step'], frame=f)
    # A star still without a place (had inside its course and kept, `held`, then out by the pause menu): at least
    # the step its record names.
    for r in hist:
        for g in list(r.get('held') or []) + list(r.get('got') or []):
            if isinstance(g, (list, tuple)) and tuple(g) not in when: when[tuple(g)] = dict(step=r['step'])
    cell = lambda c, n, name: dict(name=name, got=bool(have.get(c, 0) >> (n - 1) & 1), **when.get((c, n), {}))
    groups = [dict(name=LONG[c], short=SHORT[c], stars=[cell(c, n + 1, name) for n, name in enumerate(STAR[c])] + [cell(c, 7, '100 Coins')]) for c in range(1, 16)]
    groups += [dict(name=name, short=SHORT[c], stars=[cell(c, n + 1, x) for n, x in enumerate(names)]) for c, name, names in SECRET]
    castle_picks = {}
    for q in sc.picks:
        if q['kind'] == 'star' and q.get('who'): castle_picks.setdefault(q['who'], []).append(q['frame'])
    order = {'Toad': [24, 25, 26], 'MIPS the rabbit': [27, 28]}
    cframe = {bit: f for who, bits in order.items() for bit, f in zip(bits, castle_picks.get(who, []))}
    castle = [dict(name=name, got=bool(sc.save_flags >> bit & 1), **({'frame': cframe[bit]} if bit in cframe and sc.save_flags >> bit & 1 else {})) for bit, name in CASTLE]
    groups.append(dict(name='Castle', short='Castle', stars=castle))
    got = sum(x['got'] for g in groups for x in g['stars'])
    return dict(got=got, total=sum(len(g['stars']) for g in groups), groups=groups,
                switches=[dict(name=name, got=bool(sc.save_flags >> bit & 1)) for bit, name in SWITCHES],
                note=f"From a replay of the route from power-on: {got} of 120 stars. Hover a star for its name and when the route collects it.")


def flags_of(t):
    o = {}
    for p in (t or '').split():
        c, m = p.split(':'); o[int(c)] = int(m, 16) & 0x7f
    return o


def install(src, dst):
    tmp = dst.with_name(dst.name + '.tmp'); shutil.copyfile(src, tmp); os.replace(tmp, dst)


def main():
    tmp = tempfile.mkdtemp(prefix='viewer_publish_')
    # The state and the route it names, taken together: the campaign may finish a step in between.
    for _ in range(10):
        s = load_state(A.name); route = Path(tmp) / 'route.m64'
        # A finished campaign: its route played out through the credits (tools/seg120/credits.py).
        fin = HERE / f'logs/campaign/{A.name}/final.m64'
        shutil.copyfile(fin if s.get('done') and fin.exists() else HERE / s['route'], route)
        if load_state(A.name)['step'] == s['step']: break
    when = time.strftime('%Y-%m-%d %H:%M', time.localtime((HERE / f'logs/campaign/{A.name}/state.json').stat().st_mtime))
    live = running(A.name)
    sc = Scan(route, tmp)
    stars, keys = sc.stars[-1], sc.keys[-1]
    if (stars, keys) != (s['stars'], s['keys']): print(f"warning: the route replays to {stars} stars, {keys} keys; state.json says {s['stars']}, {s['keys']}", file=sys.stderr)
    hist = s['history']
    tail = ('The route ends here. The search is working from this state now.' if live else 'The route ends here.')
    lost = [h['used'] for h in hist if 'to' not in h and 'cp' not in h]
    keytext = {0: 'no Bowser key', 1: 'Bowser key 1', 2: 'both Bowser keys'}.get(keys, f'{keys} keys')
    desc = ("A full-game run that the search routes by itself, from power-on. A tree search locked to the castle runs until Mario enters a course. "
            "A tree search locked to that course runs until he leaves with a new star or key. Then the castle again. No TAS inputs: every frame is the search's own play. "
            f"State at {when} (step {s['step']}, {'still running' if live else 'not running'}): {stars} stars and {keytext}. "
            f"The route is {sc.total} frames ({clock(sc.total)}). {s['frames'] / 1e9:.2f} B emulator frames searched so far; "
            f"{len(lost)} of the {len(hist)} searches found no new star, key or course ({sum(lost) / 1e3:.2f} B frames). Stars: {star_list(s)}.")
    start = max(0, (sc.control or 0) - 36)
    rows = [dict(cells=[f'Campaign {A.name}: the whole run so far', f'{stars}★ {keys}🔑', sc.total, clock(sc.total), f"{s['frames'] / 1e9:.2f} B",
                        'From power-on. Each band is one level visit; its note gives the campaign step and the frames it searched. Stars and keys have their own bands.'], file=MAIN, start=start)]
    copies = [(route, VIEWER / MAIN)]
    states = {A.name: s}
    for x in EXTRAS if A.name == 't1' else []:
        src, dst = HERE / x['src'], OUT / x['dst']
        if not src.exists() and not dst.exists(): print(f"skipping {x['name']}: {x['src']} is gone", file=sys.stderr); continue
        movie = Path(tmp) / x['dst']; shutil.copyfile(src if src.exists() else dst, movie)
        h = []
        for name, last in x['hist']:
            if name not in states and (HERE / f'logs/campaign/{name}/state.json').exists(): states[name] = load_state(name)
            h += [r for r in states.get(name, {}).get('history', []) if r['step'] <= last]
        e = Scan(movie, tmp)
        enter = e.last_entry(x['open'])
        if enter is None: print(f"skipping {x['name']}: the movie never gets to {x['open']}", file=sys.stderr); continue
        search = x.get('search')
        if search is None:
            name, step = x['used']; search = next((f"{r['used']} M" for r in states.get(name, {}).get('history', []) if r['step'] == step and 'used' in r), '—')
        facts = dict(enter=enter, total=e.total, star=max((p['frame'] for p in e.picks if p['kind'] == 'star'), default=0), key=max((p['frame'] for p in e.picks if p['kind'] == 'key'), default=0))
        rows.append(dict(cells=[x['name'], f'{e.stars[-1]}★ {e.keys[-1]}🔑', e.total, clock(e.total), search, x['text'].format(**facts)],
                         file=f"obrrt2/{x['dst']}", start=max(0, enter - LEAD), regions=regions(e, h, x.get('tail', ''))))
        copies.append((movie, dst))
    entry = {'date': DATE, 'title': f"{TITLE}: {stars} star{'' if stars == 1 else 's'} so far", 'description': desc, 'file': MAIN, 'start': start, 'important': True,
             'regions': regions(sc, hist, tail),
             'stars': board(s, sc, hist),
             'progress': f'obrrt2/campaign-{A.name}-progress.json',
             'table': {'columns': ['Run', 'Ends with', 'Route frames', 'Game time', 'Searched', 'What it shows'], 'rows': rows, 'highlight': [0],
                       'note': f"Searched: emulator frames run by the tree searches that made the run. Refreshed by objective-rrt-v2/tools/seg120/viewer_publish.py; the first row is campaign {A.name} at {when}."}}
    if A.dry_run:
        print(json.dumps(entry, indent=2, ensure_ascii=False)); shutil.rmtree(tmp); return
    OUT.mkdir(exist_ok=True)
    for src, dst in copies:
        if not dst.exists() or dst.read_bytes() != Path(src).read_bytes(): install(src, dst)
    path = VIEWER / 'index.json'
    index = json.loads(path.read_text())
    mine = [i for i, e in enumerate(index) if e.get('file') == MAIN]
    if mine: index[mine[0]] = entry; index = [e for i, e in enumerate(index) if i not in mine[1:]]
    else: index.append(entry)
    t = path.with_name('index.json.tmp'); t.write_text(json.dumps(index, indent=2, ensure_ascii=False) + '\n'); os.replace(t, path)
    shutil.rmtree(tmp)
    print(f"{'updated' if mine else 'added'} \"{entry['title']}\": {stars} stars, {keys} keys, {sc.total} frames ({clock(sc.total)}), step {s['step']}; "
          f"{len(entry['regions'])} regions, {len(rows)} rows; files: {', '.join(d.name for _, d in copies)}")


main()
