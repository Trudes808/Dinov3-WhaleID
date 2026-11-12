import json, pickle, numpy as np, collections
from evalx import *
day='2025-01-02'
z=json.load(open(f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
def major(c):
    cat=c.get('catalog')
    if isinstance(cat,dict) and cat: return max(cat.items(),key=lambda kv:kv[1])[0]
    if isinstance(cat,str): return cat
    return None
print('cluster catalog field example:',[C[k].get('catalog') for k in list(C)[:4]])
maj={k:major(c) for k,c in C.items()}
animals=sorted(set(v for v in maj.values() if v)); print('catalog animals on the day',len(animals),' clusters with a catalog animal',sum(v is not None for v in maj.values()),'of',len(C))
def components(verdicts):
    par={k:k for k in C}
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for r in RECS:
        if r['verdict'] in verdicts: par[find(r['a'])]=find(r['b'])
    comp=collections.defaultdict(list)
    for k in C: comp[find(k)].append(k)
    return list(comp.values())
def stats(name,comps):
    # clusters-per-animal equivalent: components holding each catalog animal
    per=collections.defaultdict(set); mixed=0; nphot=0
    for ci,cm in enumerate(comps):
        cats=collections.Counter(maj[k] for k in cm if maj[k]); 
        for a in cats: per[a].add(ci)
        if len(cats)>1: mixed+=1
    photos=lambda cm: sum(len(C[k]['photos']) for k in cm)
    sizes=sorted([photos(cm) for cm in comps],reverse=True); tot=sum(sizes)
    print(f'{name:40s} components {len(comps):4d} | components per catalog animal {np.mean([len(v) for v in per.values()]):.2f} | holding >1 catalog animal {mixed:3d} | photos in the largest 20 components {sum(sizes[:20])/tot:.2f}, in components of 1 cluster {sum(photos(cm) for cm in comps if len(cm)==1)/tot:.2f} | singletons {sum(len(cm)==1 for cm in comps)}')
    return per
stats('safe clusters (start)',[[k] for k in C])
per=stats('+ researcher same links',components({'same'}))
stats('+ same and likely_same',components({'same','likely_same'}))
# catalog oracle on clusters with a catalog animal
print('catalog oracle: components =',len(animals),'+ clusters without catalog',sum(v is None for v in maj.values()))
# what blocks the rest: catalog-same cluster pairs within the day by researcher verdict
lab={ (min(r['a'],r['b']),max(r['a'],r['b'])):r for r in RECS}
compS=components({'same'}); cid={}
for i,cm in enumerate(compS):
    for k in cm: cid[k]=i
# for each catalog animal: its components; pairs of components of the same animal -> was any cluster pair between them labelled, and how
rows=collections.Counter(); ex=[]
for a,cs in per.items():
    cs=sorted(cs)
    for i in range(len(cs)):
        for j in range(i+1,len(cs)):
            A=[k for k in compS[cs[i]] if maj[k]==a]; B=[k for k in compS[cs[j]] if maj[k]==a]
            vs=[lab[(min(x,y),max(x,y))]['verdict'] for x in A for y in B if (min(x,y),max(x,y)) in lab]
            if not vs: rows['never shown']+=1
            elif 'different' in vs: rows['judged different']+=1
            elif 'likely_same' in vs: rows['likely same (no mark)']+=1
            elif 'unsure' in vs: rows['unsure']+=1
            else: rows['no similarity']+=1
print('catalog-same component pairs still apart after the links:',sum(rows.values()),dict(rows))
# gap bins of the labelled pairs
bins=[(-1,1.5),(1.5,60),(60,900),(900,1e9)]
for lo,hi in bins:
    c=collections.Counter(r['verdict'] for r in RECS if lo<r['gap']<=hi); tot=sum(c.values())
    print(f'gap ({lo},{hi}] s: labelled {tot:4d}  '+'  '.join(f'{k} {v} ({v/max(tot,1):.2f})' for k,v in c.most_common()))
# labelling effort
ts=json.load(open(f'/home/sat3737/NOAA_whales/Whale-Review-Web/data/{day}/pair_marks.json'))
L=[l for l in ts['labels'] if l.get('level')=='cluster' or l.get('pair_type')=='burst_link']
print('records in file',len(ts['labels']),' keys of a cluster record:',sorted(L[-1].keys()) if L else None)
