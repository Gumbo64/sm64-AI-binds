#!/usr/bin/env python3
"""The campaign's star times for sm64-viewer's live chart: stars by samples (emulator frames searched), with the
steps that took a Bowser key or pressed a cap switch.

    tools/seg120/progress_json.py [NAME] [--viewer <viewer>/frontend/public/replays]

Writes <viewer>/obrrt2/campaign-NAME-progress.json from logs/campaign/NAME/state.json alone (no replay, so it is
cheap enough to run every few seconds; tools/seg120/viewer_refresh.sh does). The viewer draws the chart itself.
Records written since 2026-11-11 carry their totals (`after`); older ones are read: a star record's `got`, a key as
an exit from a Bowser course with no star, a switch as a new `caps` value.
"""
import argparse, json, os, re, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('name', nargs='?', default='t1')
ap.add_argument('--viewer', default=str(HERE.parents[1] / 'sm64-viewer/frontend/public/replays'))
A = ap.parse_args()

BOWSER = ('BitDW', 'BitFS', 'BitS')
SWITCH = {0x2: 'Wing cap switch', 0x4: 'Metal cap switch', 0x8: 'Vanish cap switch'}


def step_times(log, year):
    '''Wall time each step ended: its last line in the campaign log ("MM-DD HH:MM:SS step N: ..."), as UTC ISO.'''
    t = {}
    for line in log.read_text(errors='replace').splitlines():
        m = re.match(r'(\d\d)-(\d\d) (\d\d:\d\d:\d\d) step (\d+)[ :]', line)
        if m: t[int(m[4])] = f'{year}-{m[1]}-{m[2]}T{m[3]}Z'
    return t


def main():
    sp = HERE / f'logs/campaign/{A.name}/state.json'
    when = step_times(HERE / f'logs/campaign/{A.name}/log.txt', time.gmtime(sp.stat().st_mtime).tm_year)
    s = json.loads(sp.read_text())
    pid = HERE / f'logs/seg120/camp-{A.name}.pid'
    try: live = pid.exists() and os.kill(int(pid.read_text()), 0) is None
    except (OSError, ValueError): live = False
    n = 0.0; stars = keys = caps = 0; events = []
    frame = 0
    points = [[0.0, 0, when.get(min(when)) if when else None, 0, 0]]
    for r in s['history']:
        n += r.get('used', 0) / 1e3                     # M frames -> B frames
        after = r.get('after')
        if after:
            st, k, c = after['stars'], after['keys'], after.get('caps', caps)
        else:
            got = r.get('got') or []
            st = max(stars, r.get('stars', stars) + len(got)) if 'to' in r or got else stars
            k = keys + 1 if 'to' in r and not got and any(b in str(r.get('zone', '')) for b in BOWSER) and r.get('to') == 'Castle' else keys
            c = r.get('caps') if r.get('caps') is not None else caps
        t = when.get(r['step'])
        if r.get('frame') is not None: frame = r['frame']
        if st != stars: points.append([round(n, 4), st, t, r['step'], frame])
        for i in range(keys + 1, k + 1): events.append(dict(kind='key', label=f'Key {i}', n=round(n, 4), step=r['step'], t=t))
        for bit, name in SWITCH.items():
            if c & bit and not caps & bit: events.append(dict(kind='switch', label=name, n=round(n, 4), step=r['step'], t=t))
        stars, keys, caps = st, k, c
    points.append([round(n, 4), stars, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(sp.stat().st_mtime)), s['step'], frame])
    out = dict(name=A.name, updated=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state_mtime=sp.stat().st_mtime, live=live,
               step=s['step'], stars=stars, keys=keys, samples=round(n, 4), unit='B emulator frames searched', fields=['samples (B frames)', 'stars', 'wall time (UTC)', 'step', 'route frame'], points=points, events=events)
    dst = Path(A.viewer) / f'obrrt2/campaign-{A.name}-progress.json'
    tmp = dst.with_name(dst.name + '.tmp'); tmp.write_text(json.dumps(out)); os.replace(tmp, dst)


main()
