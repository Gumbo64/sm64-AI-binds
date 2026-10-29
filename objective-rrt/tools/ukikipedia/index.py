import re,sys
U='ukikipedia/'
files=[l.strip() for l in open(sys.argv[1]) if l.strip()]
def entry(f):
    s=open(U+f,encoding='utf-8').read()
    title=s.split('\n',1)[0][2:].strip()
    orig=re.search(r'^Source: Ukikipedia, (\S+)',s,re.M).group(1)
    m=re.search(r'^Archived snapshot: (\S+) \((\S+)\)',s,re.M)
    body=s.split('\n---\n',1)[1]
    sent=''
    for l in body.split('\n'):
        l2=re.sub(r'\*+','',l).strip()
        if l.startswith('|') or l.startswith(':') or l.startswith('#') or l.startswith('-') or len(l2)<40: continue
        sent=re.split(r'(?<=[.!?])\s',l2)[0]; break
    if not sent: sent='Stub; infobox with the TAS/RTA records only.'
    return f"- **{title}** (Ukikipedia): original {orig} | snapshot {m.group(1)} ({m.group(2)}) | file `{U}{f}` | {sent[:300]}"
idx=open('INDEX.md',encoding='utf-8').read()
lines=idx.split('\n')
new=[]
for f in files:
    e=entry(f)
    key=f"file `{U}{f}`"
    hit=[i for i,l in enumerate(lines) if key in l]
    if hit:
        for i in hit: lines[i]=e
    else: new.append(e)
# local note entry
note="- **Star strategies, 120-star TAS** (local): file `tas/star_strategies.md` | One row per star in 2026 TAS route order: Ukikipedia's TAS strategy (or 'stub') next to our trace analysis, techniques, IL TAS WR, page link; then recurring techniques and a coverage table technique -> our objective file or missing."
if 'tas/star_strategies.md' not in idx:
    j=next(i for i,l in enumerate(lines) if l.startswith('## Ukikipedia'))
    k=j-1
    while k>0 and lines[k].strip()=='' : k-=1
    lines.insert(k+1,note)
out='\n'.join(lines).rstrip('\n')+'\n\n## Ukikipedia star pages for the 120-star TAS (2026-10-29) -> ref/ukikipedia/\n\nFetched for `tas/star_strategies.md`. The 15 per-course `100_Coin_Star_(...)` pages all redirect to `ukikipedia/100_Coin_Star.md`; `TAS` / `Tool_Assisted_Speedrun` redirect to the Tool-Assisted Speedrun stub; `120_Star_TAS` and `List_of_Stars` do not exist (404).\n\n'+'\n'.join(new)+'\n'
open('INDEX.md','w',encoding='utf-8').write(out)
print(len(new),'new entries')
