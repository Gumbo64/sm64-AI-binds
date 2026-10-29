import sys,re
# usage: extract.py file.md ... -> WR + prose of TAS-related sections + intro
for fn in sys.argv[1:]:
    s=open(fn,encoding='utf-8').read()
    print("="*20, fn.split('/')[-1])
    m=re.search(r"Archived snapshot: (\S+)",s); print("SNAP:",m.group(1) if m else None)
    m=re.search(r"\*\*TAS World Record\*\* \| ([^|]*)",s); print("WR:",m.group(1).strip() if m else None)
    body=s.split('\n---\n',1)[-1].split('### References')[0]
    lines=[l for l in body.split('\n') if l.strip() and not l.startswith('|') and not l.startswith(': *For')]
    txt='\n'.join(lines)
    # intro: first prose para
    intro=next((l for l in lines if len(l)>80 and not l.startswith('#')),'')
    print("INTRO:",intro[:300])
    secs=re.split(r'\n(?=#{2,5} )',txt)
    for sec in secs:
        h=sec.split('\n')[0]
        if re.search(r'TAS|Strateg|Route|Speedrun|120',h,re.I) or re.search(r'120 Star TAS|120-star',sec):
            print(sec[:2500]); print('--')
    for l in lines:
        if '2012' in l and '|' not in l and 'TAS' in l and l not in txt[:0]: pass
    # TAS history rows mentioning 120
    for l in body.split('\n'):
        if l.startswith('|') and re.search(r'120',l): print("HIST120:",l[:250])
