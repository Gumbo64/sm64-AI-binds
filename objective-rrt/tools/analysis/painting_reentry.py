#!/usr/bin/env python3
"""Painting re-entry after a star: for every star exit on a run's best route
(most stars, earliest), the frames from the exit node to the next painting
warp on the route, and how the search spent its attempts in between.

For each exit (the route node with the new star count, Mario back in the
castle) up to the route's next course entry:
  - route frames: exit node frame -> first route node in the next course
  - window: the attempts logged (events.jsonl, in order) from the one that
    added the exit node to the one that added the entry node
  - kin plans: attempts in the window whose b is a castle descendant of the
    exit node with the same stars (b "near" the post-star state)
  - of those, how many were Painting::Enter of the course entered next (and
    of any painting), and how many other kinds
  - painting attempts started from the exit's castle subtree, their
    successes, and the attempt index of the first one (eligibility lag)

usage: painting_reentry.py runs/<run> [...]
"""
import csv
import json
import sys
from collections import Counter, defaultdict

LEVEL_CASTLE = 6
COURSE_OF = {9: 'bob', 24: 'wf', 12: 'jrb', 5: 'ccm', 4: 'bbh', 7: 'hmc', 22: 'lll', 8: 'ssl', 23: 'ddd',
             10: 'sl', 11: 'wdw', 36: 'ttm', 13: 'thi', 14: 'ttc', 15: 'rr', 27: 'pss'}


def main(run):
    rows = {int(x['id']): x for x in csv.DictReader(open(f'{run}/nodes.csv'))}
    kids = defaultdict(list)
    for i, x in rows.items():
        kids[int(x['parent'])].append(i)
    best = max(rows.values(), key=lambda x: (int(x['stars']), -int(x['frame'])))
    path = []
    n = int(best['id'])
    while n >= 0:
        path.append(rows[n])
        n = int(rows[n]['parent'])
    path.reverse()
    events = [json.loads(l) for l in open(f'{run}/events.jsonl')]
    added_at = {e['added']: i for i, e in enumerate(events) if e.get('added') is not None}
    print(f'# {run}: best node {best["id"]}, {best["stars"]} stars at frame {best["frame"]}')
    print('exit_node exit_f stars -> entry_node course route_f | window kin_plans kin_paint_next kin_paint_any '
          'kin_top_kinds | sub_paint_tries sub_paint_succ first_paint_lag')
    for i, x in enumerate(path):
        if x['action'] != '0x1327' or int(x['level']) != LEVEL_CASTLE:
            continue
        nxt = next((y for y in path[i + 1:] if int(y['level']) != LEVEL_CASTLE), None)
        if nxt is None:
            continue
        course = COURSE_OF.get(int(nxt['level']), nxt['level'])
        ex, en = int(x['id']), int(nxt['id'])
        stars = x['stars']
        # Castle subtree of the exit with the same stars.
        sub, stack = set(), [ex]
        while stack:
            k = stack.pop()
            if int(rows[k]['level']) != LEVEL_CASTLE or rows[k]['stars'] != stars:
                continue
            sub.add(k)
            stack.extend(kids[k])
        lo, hi = added_at.get(ex, 0), added_at.get(en, len(events))
        win = events[lo + 1:hi + 1]
        kin = [e for e in win if e['b'] in sub and e['stage'] == 0]
        key = f'Painting {course}'
        kin_next = sum(e['key'] == key for e in kin)
        kin_any = sum(e['kind'] == 'Painting::Enter' for e in kin)
        top = Counter(e['kind'] for e in kin).most_common(4)
        sp = [j for j, e in enumerate(win) if e['start'] in sub and e['kind'] == 'Painting::Enter' and e['stage'] == 0]
        succ = sum(win[j]['success'] for j in sp)
        print(f'{ex:>6} {x["frame"]:>6} {stars} -> {en:>6} {course:<4} {int(nxt["frame"]) - int(x["frame"]):>5} | '
              f'{len(win):>5} {len(kin):>5} {kin_next:>4} {kin_any:>4} '
              f'{",".join(f"{k}:{v}" for k, v in top):<70} | {len(sp):>4} {succ:>4} {sp[0] if sp else "-"}')


if __name__ == '__main__':
    for r in sys.argv[1:]:
        main(r.rstrip('/'))
        print()
