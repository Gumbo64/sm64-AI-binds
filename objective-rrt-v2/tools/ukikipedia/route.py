import csv,re,sys,math
spots=[]
for l in open('objectives/src/extra/missions/table.rs'):
    m=re.search(r'level: LEVEL_(\w+), area: (\d+).*pos: \[([-\d.]+), ([-\d.]+), ([-\d.]+)\].*name: "([^"]+)"',l)
    if m: spots.append((m.group(1),float(m.group(3)),float(m.group(4)),float(m.group(5)),m.group(6)))
lvl={'1':'?'}
prev=None; out=[]
with open('ref/tas/dumps/full-120star-2026-U.trace.csv') as f:
    r=csv.DictReader(f)
    last=None
    for row in r:
        st=int(row['stars'])
        if prev is not None and st>prev:
            out.append((st,int(row['frame']),row['level'],row['area'],float(last['x']),float(last['y']),float(last['z']),last['act_name']))
        prev=st; last=row
for st,fr,lv,ar,x,y,z,act in out:
    best=None
    for s in spots:
        d=math.dist((x,y,z),s[1:4])
        if best is None or d<best[0]: best=(d,s[4],s[0])
    nm=best[1] if best and best[0]<700 else ''
    print(f"{st}\t{fr}\t{fr/30:.0f}s\t{lv}/{ar}\t({x:.0f},{y:.0f},{z:.0f})\t{act}\t{nm}\t{best[0]:.0f}" )
