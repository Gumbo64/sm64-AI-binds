#!/usr/bin/env python3
"""conv.py in.html final_archive_url out.md -- archived Ukikipedia HTML -> markdown."""
import sys, re, html, datetime
from html.parser import HTMLParser
SKIP = {"navbox","toc","mw-editsection","printfooter","catlinks","noprint","mw-jump-link","navbar"}
VOID = {"br","img","hr","meta","link","input","wbr","source"}
class Conv(HTMLParser):
    def __init__(s):
        super().__init__(convert_charrefs=True); s.buf=[]; s.skip=0; s.tables=[]; s.lists=[]; s.href=[]; s.pre=0
    def w(s,x):
        if s.skip: return
        if s.tables and s.tables[-1]["cell"] is not None: s.tables[-1]["cell"].append(x)
        else: s.buf.append(x)
    def handle_starttag(s,tag,attrs):
        a=dict(attrs); cls=set((a.get("class") or "").split()); void=tag in VOID
        if s.skip:
            if not void: s.skip+=1
            return
        if (cls&SKIP) or tag in ("style","script") or a.get("id")=="toc" or a.get("style","").replace(" ","").find("display:none")>=0:
            if not void: s.skip=1
            return
        incell = s.tables and s.tables[-1]["cell"] is not None
        if tag[0]=="h" and tag[1:].isdigit(): s.w("\n\n"+"#"*max(int(tag[1])+1,3)+" ")
        elif tag=="p": s.w(" " if incell else "\n\n")
        elif tag=="br": s.w(" " if incell else "\n")
        elif tag in ("b","strong"): s.w("**")
        elif tag in ("i","em"): s.w("*")
        elif tag in ("ul","ol"): s.lists.append([tag,0]); s.w("\n")
        elif tag=="li":
            d=len(s.lists)
            if s.lists and s.lists[-1][0]=="ol": s.lists[-1][1]+=1; m=f"{s.lists[-1][1]}. "
            else: m="- "
            s.w(" " if incell else "\n"+"  "*max(d-1,0)+m)
        elif tag=="dd": s.w("\n: ")
        elif tag=="pre": s.w("\n```\n"); s.pre+=1
        elif tag=="code": s.w("`")
        elif tag=="table": s.tables.append({"rows":[],"cell":None,"caption":[]})
        elif tag=="tr" and s.tables: s.tables[-1]["rows"].append([])
        elif tag in ("td","th") and s.tables:
            t=s.tables[-1]
            if not t["rows"]: t["rows"].append([])
            t["cell"]=[]; t["rows"][-1].append(t["cell"])
            try: span=int(a.get("colspan","1"))
            except ValueError: span=1
            for _ in range(min(span,10)-1): t["rows"][-1].append([])
        elif tag=="caption" and s.tables: s.tables[-1]["cell"]=s.tables[-1]["caption"]
        elif tag=="a": s.href.append(a.get("href") or "")
        elif tag=="img" and a.get("alt") and not incell: pass
    def handle_endtag(s,tag):
        if s.skip: s.skip-=1; return
        if tag[0]=="h" and tag[1:].isdigit(): s.w("\n")
        elif tag in ("b","strong"): s.w("**")
        elif tag in ("i","em"): s.w("*")
        elif tag in ("ul","ol"):
            if s.lists: s.lists.pop()
            s.w("\n")
        elif tag=="pre": s.w("\n```\n"); s.pre-=1
        elif tag=="code": s.w("`")
        elif tag in ("td","th","caption") and s.tables: s.tables[-1]["cell"]=None
        elif tag=="table" and s.tables: s.w(render(s.tables.pop()))
        elif tag=="a":
            h=s.href.pop() if s.href else ""
            u=re.sub(r"^(https?://web\.archive\.org)?/web/\d+[a-z_]*/","",h)
            if u.startswith("http") and "ukikipedia.net" not in u: s.w(f" (<{u}>)")
    def handle_data(s,d):
        if s.skip: return
        if not s.pre: d=re.sub(r"\s+"," ",d)
        s.w(d)
def cc(c):
    x=re.sub(r"\s+"," ","".join(c)).strip().replace("|","\\|"); return re.sub(r"\*\*\s*\*\*","",x)
def render(t):
    rows=[[cc(c) for c in r] for r in t["rows"] if r]
    if not rows: return ""
    n=max(len(r) for r in rows); rows=[r+[""]*(n-len(r)) for r in rows]
    o=["\n\n"]; cap=cc(t["caption"])
    if cap: o.append(f"  {cap}\n\n")
    o.append("| "+" | ".join(rows[0])+" |\n|"+"---|"*n+"\n")
    for r in rows[1:]: o.append("| "+" | ".join(r)+" |\n")
    return "".join(o)+"\n"
src,final,outp=sys.argv[1:4]
s=open(src,encoding="utf-8",errors="replace").read()
m=re.search(r'<h1[^>]*id="firstHeading"[^>]*>(.*?)</h1>',s,re.S)
title=html.unescape(re.sub(r"<[^>]+>","",m.group(1))).strip() if m else None
if not title:
    t=re.search(r"<title>(.*?)</title>",s,re.S); title=html.unescape(t.group(1)).split(" - ")[0].strip() if t else "?"
i=s.find('<div class="mw-parser-output">'); j=s.find('<div class="printfooter"',i)
body=re.sub(r"<!--.*?-->","",s[i:j if j>0 else len(s)],flags=re.S)
c=Conv(); c.feed(body); c.close()
md="".join(c.buf); md=re.sub(r"[ \t]+\n","\n",md); md=re.sub(r"\n{3,}","\n\n",md).strip()
ts=re.search(r"web\.archive\.org/web/(\d{14})",final).group(1); d=f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
orig=re.sub(r"^https?://web\.archive\.org/web/\d+[a-z_]*/","",final)
rs=re.search(r'Redirected from[^<]*<a[^>]*>([^<]*)</a>',s)
red=f"Redirected from: {html.unescape(rs.group(1))}\n" if rs else ""
hdr=(f"# {title}\n\nSource: Ukikipedia, {orig}\nArchived snapshot: {final} ({d})\n{red}"
     f"Retrieved via the Internet Archive Wayback Machine on {datetime.date.today().isoformat()}. Content is CC BY-SA (Ukikipedia); navigation stripped.\n\n---\n\n")
open(outp,"w",encoding="utf-8").write(hdr+md+"\n"); print(title)
