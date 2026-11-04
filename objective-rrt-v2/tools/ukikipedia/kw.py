import sys,re
KW=re.compile(r"BLJ|hyperspeed|clip|pause|RNG|camera|cannon|C-?up|global timer|PU\b|parallel universe|upwarp|downwarp|wall ?kick|shell|ledge grab|lava boost|death|120|TAS|slide|dive|wing cap|metal cap|vanish|pole|grind|misalign",re.I)
for fn in sys.argv[1:]:
    s=open(fn,encoding='utf-8').read()
    body=s.split('\n---\n',1)[-1]
    body=re.split(r'### References',body)[0]
    out=[]
    for l in body.split('\n'):
        if l.startswith('|'):
            cells=[c.strip() for c in l.strip('|').split('|')]
            # history row: date | time | ... | notes
            if len(cells)>=4 and re.search(r'\d{4}',cells[0]) and cells[-1] and len(cells[-1])>12: out.append("ROW: "+" | ".join(cells)[:300])
        elif KW.search(l) and len(l)>60 and not l.startswith(': *For') and 'A press' not in l and 'ABC' not in l:
            out.append("TXT: "+l[:600])
    if out: print("=== "+fn); print("\n".join(out[-12:]))
