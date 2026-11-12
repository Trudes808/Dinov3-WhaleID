import json, collections, sys, datetime
ROOT='/home/sat3737/NOAA_whales'; DAY='2025-01-02'
z=json.load(open(f'{ROOT}/Dinov3-WhaleID/experiments/days_all/{DAY}/safe/clusters.json'))
C={c['id']:c for c in z['clusters']}
r=json.load(open(f'{ROOT}/Whale-Review-Web/data/{DAY}/pair_marks.json'))
L=[x for x in r['labels'] if x.get('level')=='cluster']
def major(c):
    e=sorted((c.get('catalog') or {}).items(), key=lambda kv:-kv[1]); return e[0][0] if e else None
def gap(a,b):
    A,B=C[a],C[b]; return max(0.0, max(A['start'],B['start'])-min(A['end'],B['end']))
sh=lambda k: k.split('-')[-1]
for verdict in ('same','likely_same'):
    print('=====',verdict,'with catalog different')
    for x in L:
        a,b=x['set_a'],x['set_b']; ca,cb=major(C[a]),major(C[b])
        if x['verdict']==verdict and ca and cb and ca!=cb:
            A,B=C[a],C[b]
            print(f"{a:4d}|{b:4d} gap {gap(a,b):6.1f}s  A[{len(A['photos'])}ph multi{A.get('n_multi')}] {dict((k[-6:],v) for k,v in A['catalog'].items())}  B[{len(B['photos'])}ph multi{B.get('n_multi')}] {dict((k[-6:],v) for k,v in B['catalog'].items())}  keys {sh(x['key_a'])} {sh(x['key_b'])} cue {x['cue']} catwrong {x['catalog_wrong']} score {x.get('model_score')} rec.catalog {x.get('catalog')} nreg {len(x['regions'])}")
print(collections.Counter(x.get('catalog') for x in L))
print('cluster keys', [k for k in C[5].keys()])
