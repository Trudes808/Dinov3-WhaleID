import json, collections, statistics as st, sys
import numpy as np
ROOT='/home/sat3737/NOAA_whales'
DAY=sys.argv[1] if len(sys.argv)>1 else '2025-01-02'
z=json.load(open(f'{ROOT}/Dinov3-WhaleID/experiments/days_all/{DAY}/safe/clusters.json'))
C={c['id']:c for c in z['clusters']}
r=json.load(open(f'{ROOT}/Whale-Review-Web/data/{DAY}/pair_marks.json'))
L=[x for x in r['labels'] if x.get('level')=='cluster']
print('cluster records',len(L),'clusters',len(C), 'photos', sum(len(c['photos']) for c in C.values()))
# last record per pair wins
last={}
for x in L:
    k=tuple(sorted((x['set_a'],x['set_b'])))
    last[k]=x
print('unique pairs',len(last))
V=collections.Counter(x['verdict'] for x in last.values()); print(V)
def major(c):
    e=sorted((c.get('catalog') or {}).items(), key=lambda kv:-kv[1]); return e[0][0] if e else None
def pure(c): return len(c.get('catalog') or {})<=1
cat={i:major(c) for i,c in C.items()}
print('clusters with catalog', sum(v is not None for v in cat.values()), 'impure clusters', sum(not pure(c) for c in C.values()), 'catalog animals', len({v for v in cat.values() if v}))
def gap(a,b):
    A,B=C[a],C[b]; return max(0.0, max(A['start'],B['start'])-min(A['end'],B['end']))
tab=collections.defaultdict(collections.Counter)
gaps=collections.defaultdict(list)
for (a,b),x in last.items():
    ca,cb=cat.get(a),cat.get(b)
    rel='unknown' if ca is None or cb is None else 'same' if ca==cb else 'different'
    tab[x['verdict']][rel]+=1
    gaps[x['verdict']].append(gap(a,b))
print('\nverdict x catalog relation (majority animal of each cluster)')
for v in ('same','likely_same','no_similarity','different','unsure','skip'):
    if v in tab: print(f'  {v:14s}', dict(tab[v]), ' gap s: median %.1f p25 %.1f p75 %.1f max %.0f'%(np.median(gaps[v]),np.percentile(gaps[v],25),np.percentile(gaps[v],75),max(gaps[v])), ' >1.5s:',sum(g>1.5 for g in gaps[v]), ' >60s:',sum(g>60 for g in gaps[v]), ' >300s:', sum(g>300 for g in gaps[v]))
# union-find over same links
def uf_components(pairs, ids):
    p={i:i for i in ids}
    def f(i):
        while p[i]!=i:
            p[i]=p[p[i]]; i=p[i]
        return i
    for a,b in pairs:
        ra,rb=f(a),f(b)
        if ra!=rb: p[ra]=rb
    comp=collections.defaultdict(list)
    for i in ids: comp[f(i)].append(i)
    return list(comp.values())
for name,vs in (('same',('same',)),('same+likely',('same','likely_same'))):
    pairs=[k for k,x in last.items() if x['verdict'] in vs]
    comps=uf_components(pairs,list(C))
    multi=[c for c in comps if len(c)>1]
    # purity vs catalog
    imp=0; impl=[]
    for c in comps:
        an=collections.Counter()
        for i in c:
            for k,v in (C[i].get('catalog') or {}).items(): an[k]+=v
        if len(an)>1:
            imp+=1; impl.append((len(c),dict(an)))
    # restrict to clusters with catalog
    withcat=[i for i in C if cat[i] is not None]
    comps_cat=[[i for i in c if cat[i] is not None] for c in comps]; comps_cat=[c for c in comps_cat if c]
    n_an=len({cat[i] for i in withcat})
    photos=lambda ids: sum(len(C[i]['photos']) for i in ids)
    print(f'\n[{name}] links {len(pairs)} -> components {len(comps)} (multi {len(multi)}, clusters in multi {sum(len(c) for c in multi)}, photos in multi {sum(photos(c) for c in multi)}), singletons {len(comps)-len(multi)}')
    print(f'   catalog-labelled clusters {len(withcat)} in {len(comps_cat)} components for {n_an} animals = {len(comps_cat)/n_an:.2f} per animal (before: {len(withcat)/n_an:.2f})')
    print(f'   components holding >1 catalog animal: {imp}')
    for s,an in sorted(impl, key=lambda t:-t[0])[:12]: print('      ',s,an)
    # per-animal fragment count
    frag=collections.Counter()
    for c in comps_cat:
        an=collections.Counter(cat[i] for i in c); frag[an.most_common(1)[0][0]]+=1
    print('   fragments per animal:', sorted(frag.values(), reverse=True))
# contradictions: different verdict inside a same-component
pairs=[k for k,x in last.items() if x['verdict']=='same']
comps=uf_components(pairs,list(C)); cid={i:n for n,c in enumerate(comps) for i in c}
print('\n"different" verdicts inside one same-component:', sum(cid[a]==cid[b] for (a,b),x in last.items() if x['verdict']=='different'))
print('"no_similarity" pairs already joined through other same links:', sum(cid[a]==cid[b] for (a,b),x in last.items() if x['verdict']=='no_similarity'), 'of', V['no_similarity'])
print('"likely_same" pairs joined through same links:', sum(cid[a]==cid[b] for (a,b),x in last.items() if x['verdict']=='likely_same'), 'of', V['likely_same'])
# regions and cues
same=[x for x in last.values() if x['verdict']=='same']
nreg=collections.Counter(len(x['regions']) for x in same); print('\nregion pairs per same link', dict(nreg), 'total', sum(len(x['regions']) for x in same))
cues=collections.Counter(c for x in same for c in x.get('cue') or []); print('cues', dict(cues)); print('cue combos', collections.Counter(tuple(sorted(x.get('cue') or [])) for x in same).most_common(12))
print('no-mark cues', collections.Counter(tuple(sorted(x.get('cue') or [])) for x in last.values() if x['verdict']=='no_similarity').most_common(6))
print('notes:', [x['note'] for x in last.values() if x.get('note')][:40])
print('catalog_wrong ticks:', sum(bool(x.get('catalog_wrong')) for x in last.values()))
# suggestion coverage
sg=z['suggestions']; done=0; tot=0; gapdone=0; gaptot=0
for g in sg:
    k=tuple(sorted((g['a'],g['b']))); gp=gap(g['a'],g['b']); tot+=1; d=k in last; done+=d
    if gp>1.5: gaptot+=1; gapdone+=d
print(f'\nsuggestions labelled {done}/{tot}; across gap {gapdone}/{gaptot}; labelled pairs not in suggestions {sum(1 for k in last if k not in {tuple(sorted((g["a"],g["b"]))) for g in sg})}')
# catalog-same suggestions across gap: labelled?
cs=[g for g in sg if cat.get(g['a']) and cat.get(g['a'])==cat.get(g['b']) and gap(g['a'],g['b'])>1.5]
print('catalog-same across-gap suggestions', len(cs), 'labelled', sum(tuple(sorted((g['a'],g['b']))) in last for g in cs))
