import sys,re
for fn in sys.argv[1:]:
    s=open(fn,encoding='utf-8').read().split('\n---\n',1)[-1].split('### References')[0]
    rows=[]; intable=False
    for l in s.split('\n'):
        if l.startswith('| ') and re.search(r'\bDate\b',l,re.I) and re.search(r'Time|Record',l,re.I): intable=True; rows.append(('H',l)); continue
        if intable and l.startswith('|'):
            if l.startswith('|---'): continue
            rows.append(('R',l))
        elif intable and not l.startswith('|'): intable=False
    r=[x[1] for x in rows if x[0]=='R' and re.search(r'\d',x[1])]
    sel=[x for x in r if re.search(r'120',x)]+r[-2:]
    print("=== "+fn)
    for x in dict.fromkeys(sel): print("  "+re.sub(r'\s+',' ',x)[:260])
