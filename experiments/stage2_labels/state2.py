import json, pickle, numpy as np, collections, itertools
from evalx import *
day='2025-01-02'
z=json.load(open(f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
def major(c):
    e=sorted((c.get('catalog') or {}).items(), key=lambda kv:-kv[1]); return e[0][0] if e else None
maj={k:major(c) for k,c in C.items()}; animals=sorted(set(v for v in maj.values() if v))
def gap(a,b):
    A,B=C[a],C[b]; return max(0.0,max(A['start'],B['start'])-min(A['end'],B['end']))
def uf(pairs):
    par={k:k for k in C}
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for a,b in pairs: par[find(a)]=find(b)
    comp=collections.defaultdict(list)
    for k in C: comp[find(k)].append(k)
    return list(comp.values())
def per_animal(comps):
    per=collections.defaultdict(set)
    for ci,cm in enumerate(comps):
        for k in cm:
            if maj[k]: per[maj[k]].add(ci)
    return np.mean([len(v) for v in per.values()]), per
cat_ids=[k for k in C if maj[k]]
print('animals',len(animals),'clusters with catalog',len(cat_ids))
print('pieces per catalog animal: start %.2f'%per_animal([[k] for k in C])[0])
same=[(r['a'],r['b']) for r in RECS if r['verdict']=='same']
print('after researcher same links %.2f'%per_animal(uf(same))[0])
for w in (1.5,60,300,900,3600,1e9):
    orc=[(a,b) for a,b in itertools.combinations(cat_ids,2) if maj[a]==maj[b] and gap(a,b)<=w]
    print(f'catalog oracle joining every catalog-same pair within {w:>8} s: pieces per animal {per_animal(uf(orc))[0]:.2f}   (pairs {len(orc)})')
# researcher links + oracle beyond: what remains after the researcher within 15 min
compS=uf(same); m,per=per_animal(compS)
cid={k:i for i,cm in enumerate(compS) for k in cm}
gaps=[];
for a,cs in per.items():
    cs=sorted(cs)
    for i in range(len(cs)):
        for j in range(i+1,len(cs)):
            A=[k for k in compS[cs[i]] if maj[k]==a]; B=[k for k in compS[cs[j]] if maj[k]==a]
            gaps.append(min(gap(x,y) for x in A for y in B))
gaps=np.array(gaps); print('catalog-same component pairs still apart',len(gaps),' closest approach <=60 s',(gaps<=60).sum(),' 60 s-15 min',((gaps>60)&(gaps<=900)).sum(),' >15 min',(gaps>900).sum())
# the day's timeline per animal: number of separate sighting runs (gap > 15 min between consecutive clusters of the animal)
runs=[]
for a in animals:
    cl=sorted([C[k] for k in cat_ids if maj[k]==a],key=lambda c:c['start']); n=1; end=cl[0]['end']
    for c in cl[1:]:
        if c['start']-end>900: n+=1
        end=max(end,c['end'])
    runs.append(n)
print('sighting runs per animal (new run after 15 min without a photo): mean %.2f, median %d, max %d'%(np.mean(runs),np.median(runs),max(runs)))
# labelled share of candidate pairs within 15 min
allp=[(a,b) for a,b in itertools.combinations(sorted(C),2) if gap(a,b)<=900]
lab={(min(r['a'],r['b']),max(r['a'],r['b'])) for r in RECS}
catsame=[p for p in allp if maj[p[0]] and maj[p[0]]==maj[p[1]]]
print('cluster pairs within 15 min',len(allp),' labelled',len(lab),' catalog-same among them',len(catsame),' catalog-same labelled',sum(p in lab for p in catsame))
sug={(min(s['a'],s['b']),max(s['a'],s['b'])) for s in z['suggestions']}
print('suggestions',len(sug),' labelled suggestions',len(sug&lab),' catalog-same pairs that are suggestions',sum(p in sug for p in catsame))
# within-component redundancy: catalog-same pairs within 15 min already inside one researcher component
print('catalog-same pairs within 15 min already inside one researcher component',sum(cid[a]==cid[b] for a,b in catsame),'of',len(catsame))
print('---- queue ceiling ----')
rest=[p for p in sug if p not in lab]
rest_x=[p for p in rest if cid[p[0]]!=cid[p[1]]]
print('unlabelled suggestions',len(rest),' between different researcher components',len(rest_x),' of those catalog-same',sum(1 for a,b in rest_x if maj[a] and maj[a]==maj[b]),' catalog-different',sum(1 for a,b in rest_x if maj[a] and maj[b] and maj[a]!=maj[b]))
print('ceiling: researcher links + every catalog-same unlabelled suggestion: pieces per animal %.2f'%per_animal(uf(same+[(a,b) for a,b in rest_x if maj[a] and maj[a]==maj[b]]))[0])
print('ceiling: every catalog-same suggestion alone: %.2f'%per_animal(uf([(a,b) for a,b in sug if maj[a] and maj[a]==maj[b]]))[0])
# component-level candidates within 15 min not covered by any suggestion
comp_pairs=collections.defaultdict(list)
for a,b in allp:
    if cid[a]!=cid[b]: comp_pairs[(min(cid[a],cid[b]),max(cid[a],cid[b]))].append((a,b))
def cmaj(ci):
    c=collections.Counter(maj[k] for k in compS[ci] if maj[k]); return c.most_common(1)[0][0] if c else None
cs=[(p,cmaj(p[0]),cmaj(p[1])) for p in comp_pairs]
same_cp=[p for p,x,y in cs if x and x==y]
print('component pairs within 15 min',len(comp_pairs),' catalog-same',len(same_cp),' of those with an unlabelled suggestion',sum(any(q in sug and q not in lab for q in comp_pairs[p]) for p in same_cp),' with a labelled pair (no link made)',sum(any(q in lab for q in comp_pairs[p]) for p in same_cp))
