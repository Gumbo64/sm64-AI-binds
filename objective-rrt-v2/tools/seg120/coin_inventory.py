#!/usr/bin/env python3
'''Static per-course, per-area coin inventory from the decomp (levels/<lvl>/script.c, areas/<n>/macro.inc.c and
collision.inc.c special objects). Writes coin_inventory.json next to this file (run: python3 coin_inventory.py).

Per area (what a visit to that area can give; the coin counter is per level visit, not per area, and a collected
coin stays collected when Mario leaves and re-enters the area: coin.inc.c sets the respawn bits, object_list_processor.c
unload_deactivated_objects_in_list; numCoins is reset only on level entry, level_update.c):
  fixed  = yellow (1) + formations (5 line, 8 ring/arrow: coin.inc.c spawn_coin_in_formation) + red (2) + blue (5: moving
           blue coins and hidden blue coins behind a switch) + special-object coins + boxes (! box type 4/5/6: 1/3/10)
  loot   = coins that enemies drop (hitbox numLootCoins of goombas etc.); only if the enemy is beaten
  total  = fixed + loot. `warps` = area -> areas its WARP_NODEs lead to (same level).
'''
import json, os, re, sys

PORT = '/workspaces/sm64-combined/sm64-binds/sm64-port'
COURSES = {1: 'bob', 2: 'wf', 3: 'jrb', 4: 'ccm', 5: 'bbh', 6: 'hmc', 7: 'lll', 8: 'ssl', 9: 'ddd', 10: 'sl', 11: 'wdw', 12: 'ttm',
           13: 'thi', 14: 'ttc', 15: 'rr'}
# Enemies' fixed loot (ObjectHitbox numLootCoins in src/game/behaviors/*.inc.c).
LOOT = {'bhvGoomba': 1, 'bhvFlyGuy': 2, 'bhvSpindrift': 3, 'bhvFirePiranhaPlant': 1, 'bhvSkeeter': 3, 'bhvMrBlizzard': 3,
        'bhvSnufit': 2, 'bhvEnemyLakitu': 5, 'bhvMrI': 5, 'bhvSwoop': 1, 'bhvScuttlebug': 3, 'bhvJumpingBox': 5, 'bhvKoopaShell': 1,
        # SET_INT(oNumLootCoins, n) in data/behavior_data.c, a pokey's head (blue coin: pokey.inc.c), a small box (3: breakable_box_small.inc.c)
        'bhvChuckya': 5, 'bhvPiranhaPlant': 5, 'bhvSmallWhomp': 5, 'bhvWoodenPost': 5, 'bhvPokey': 5, 'bhvBreakableBoxSmall': 3}
SPECIAL_COIN = {'special_yellow_coin': 1, 'special_yellow_coin_2': 1, 'special_moving_blue_coin': 5}
BOX = {4: 1, 5: 3, 6: 10}


def macro_presets():
    names = re.findall(r'^\s*(macro_\w+),?\s*$', open(f'{PORT}/include/macro_preset_names.h').read(), re.M)
    ents = re.findall(r'^\s*\{(bhv\w+),\s*\w+,\s*([^}]*)\}', open(f'{PORT}/include/macro_presets.h').read(), re.M)
    assert len(names) == len(ents), (len(names), len(ents))
    flags = {'COIN_FORMATION_FLAG_VERTICAL': 1, 'COIN_FORMATION_FLAG_RING': 2, 'COIN_FORMATION_FLAG_ARROW': 4, 'COIN_FORMATION_FLAG_FLYING': 16}
    out = {}
    for n, (b, p) in zip(names, ents):
        p = p.split('//')[0].strip()
        v = 0
        for t in p.split('|'):
            t = t.strip()
            v |= flags[t] if t in flags else (int(t, 0) if t else 0)
        out[n] = (b, v)
    return out


def item(bhv, p2, params, d):
    '''Add one object's coins to the area dict d.'''
    t = (params >> 16) & 0xFF
    if bhv in ('bhvYellowCoin', 'bhvOneCoin'): d['yellow'] += 1
    elif bhv == 'bhvCoinFormation': d['formation'] += 5 if t & 7 < 2 else 8; d['formations'] += 1
    elif bhv == 'bhvRedCoin': d['red'] += 2; d['red_n'] += 1
    elif bhv in ('bhvMovingBlueCoin', 'bhvBlueCoinSliding'): d['blue'] += 5
    elif bhv == 'bhvHiddenBlueCoin': d['blue_hidden'] += 5
    elif bhv == 'bhvExclamationBox' and t in BOX: d['box'] += BOX[t]
    elif bhv == 'bhvBreakableBox' and t: d['loot'] += 3
    elif bhv in LOOT: d['loot'] += LOOT[bhv]


def new_area():
    return dict(yellow=0, formation=0, formations=0, red=0, red_n=0, blue=0, blue_hidden=0, box=0, special=0, loot=0, warps=[])


def parse_level(lvl, presets):
    base = f'{PORT}/levels/{lvl}'
    src = open(f'{base}/script.c').read()
    funcs = {m[1]: m[2] for m in re.finditer(r'static const LevelScript (script_func_local_\d+)\[\] = \{(.*?)\n\};', src, re.S)}
    areas = {}
    for m in re.finditer(r'^\s*AREA\(\s*/\*index\*/\s*(\d+),.*?^\s*END_AREA\(\)', src, re.S | re.M):
        n = int(m[1]); body = m[0]; d = new_area(); areas[n] = d
        text = body
        for f in re.findall(r'JUMP_LINK\((script_func_local_\d+)\)', body): text += funcs.get(f, '')
        for o in re.finditer(r'OBJECT(?:_WITH_ACTS)?\(.*?/\*behParam\*/\s*(0x[0-9a-fA-F]+),\s*/\*beh\*/\s*(\w+)', text):
            item(o[2], 0, int(o[1], 16), d)
        for w in re.finditer(r'WARP_NODE\(.*?/\*destLevel\*/\s*(\w+),\s*/\*destArea\*/\s*(0x[0-9a-fA-F]+)', body):
            if w[1] == f'LEVEL_{lvl.upper()}' and int(w[2], 16) != n and int(w[2], 16) not in d['warps']: d['warps'].append(int(w[2], 16))
        # Instant warps (floors that move Mario to another area: DDD's sub, SSL's Eyerok room, TTM's slide).
        for w in re.finditer(r'INSTANT_WARP\(\s*/\*index\*/\s*\d+,\s*/\*destArea\*/\s*(\d+)', body):
            if int(w[1]) != n and int(w[1]) not in d['warps']: d['warps'].append(int(w[1]))
        mf = f'{base}/areas/{n}/macro.inc.c'
        if os.path.exists(mf):
            for o in re.finditer(r'MACRO_OBJECT(_WITH_BEH_PARAM)?\(\s*/\*preset\*/\s*(\w+),.*?(?:/\*behParam\*/\s*(\w+))?\)\s*,', open(mf).read()):
                pre = o[2]
                if pre not in presets: continue
                b, pv = presets[pre]
                par = 0
                if o[3]:
                    try: par = int(o[3], 0)
                    except ValueError: par = 0
                if pv != 0: par = (par & 0xFF00) + (pv & 0xFF)
                item(b, 0, ((par & 0xFF) << 16) + (par & 0xFF00), d)
        cf = f'{base}/areas/{n}/collision.inc.c'
        if os.path.exists(cf):
            for o in re.finditer(r'SPECIAL_OBJECT\w*\(\s*/\*preset\*/\s*(\w+)', open(cf).read()):
                d['special'] += SPECIAL_COIN.get(o[1], 0)
    for d in areas.values():
        d['fixed'] = d['yellow'] + d['formation'] + d['red'] + d['blue'] + d['blue_hidden'] + d['box'] + d['special']
        d['total'] = d['fixed'] + d['loot']
    return areas


def main():
    presets = macro_presets()
    inv = {}
    for c, lvl in COURSES.items():
        areas = parse_level(lvl, presets)
        inv[str(c)] = dict(level=lvl, areas={str(k): v for k, v in sorted(areas.items())},
                           total=sum(a['total'] for a in areas.values()), fixed=sum(a['fixed'] for a in areas.values()))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coin_inventory.json')
    json.dump(inv, open(out, 'w'), indent=1)
    for c, v in inv.items():
        print(c, v['level'], 'fixed', v['fixed'], 'total', v['total'], {k: (a['fixed'], a['total']) for k, a in v['areas'].items()})


if __name__ == '__main__':
    main()
