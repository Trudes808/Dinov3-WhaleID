"""PCA, t-SNE (the page's two layouts) and UMAP in 3D on the day's fingerprints: what does each picture keep?
  neighbours   share of a point's 10 closest look-alikes (512 numbers, cosine) that are among its 10 closest in the picture
  animals      of a point's 5 closest in the picture from OTHER stacks, the share with its catalogue animal (in the 512
               numbers themselves: the ceiling the picture can reach)
  far apart    rank correlation of the distances in the picture with the distances between fingerprints (all pairs)"""
import glob, os, sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
import umap
def knn(Dm, k, block=None):
    Dm = Dm.copy(); np.fill_diagonal(Dm, np.inf)
    if block is not None: Dm[block] = np.inf
    return np.argsort(Dm, 1)[:, :k]
def measure(E, Y, stack, cat):
    DE = squareform(pdist(E, 'cosine')); DY = squareform(pdist(Y)); n = len(E)
    a, b = knn(DE, 10), knn(DY, 10); nb = np.mean([len(set(a[i]) & set(b[i])) / 10 for i in range(n)])
    same_stack = stack[:, None] == stack[None, :]; lab = cat != ''
    def purity(Dm):
        nn = knn(Dm, 5, same_stack); v = [np.mean(cat[nn[i]] == cat[i]) for i in range(n) if lab[i]]; return float(np.mean(v)) if v else float('nan')
    iu = np.triu_indices(n, 1); rho = spearmanr(DE[iu], DY[iu]).statistic
    return nb, purity(DY), purity(DE), rho
rows = []; tm = {'tsne': 0.0, 'umap': 0.0}
for f in sorted(glob.glob('fp/*.npz')):
    z = np.load(f); E = z['E']; n = len(E); day, part = os.path.basename(f)[:-4].split('_'); out = {}
    out['PCA (page)'] = z['pca']; out['t-SNE (page)'] = z['tsne']
    for name, kw in (('UMAP 15 neighbours', dict(n_neighbors=15, min_dist=0.1)), ('UMAP 30 neighbours', dict(n_neighbors=30, min_dist=0.1)), ('UMAP 15, spread out', dict(n_neighbors=15, min_dist=0.5))):
        t = time.time(); out[name] = umap.UMAP(n_components=3, metric='cosine', random_state=0, **{k: (min(v, n - 1) if k == 'n_neighbors' else v) for k, v in kw.items()}).fit_transform(E); tm['umap'] += time.time() - t
    t = time.time(); TSNE(n_components=3, perplexity=float(min(30, max(5, (n - 1) // 3))), init='pca', random_state=0, metric='cosine').fit_transform(E); tm['tsne'] += time.time() - t
    # stability: the same layout with another seed, how much of each point's picture-neighbourhood stays
    for name, Y in out.items():
        nb, pur, ceil, rho = measure(E, Y, z['stack'], z['cat']); rows.append(dict(day=day, part=part, n=n, name=name, nb=nb, pur=pur, ceil=ceil, rho=rho))
    Y2 = umap.UMAP(n_components=3, metric='cosine', random_state=1, n_neighbors=min(15, n - 1), min_dist=0.1).fit_transform(E); T2 = TSNE(n_components=3, perplexity=float(min(30, max(5, (n - 1) // 3))), init='pca', random_state=1, metric='cosine').fit_transform(E)
    st = lambda A, B: float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(knn(squareform(pdist(A)), 10), knn(squareform(pdist(B)), 10))]))
    rows.append(dict(day=day, part=part, n=n, name='_stab', umap=st(out['UMAP 15 neighbours'], Y2), tsne=st(z['tsne'], T2)))
    print(day, part, n, 'done', flush=True)
names = ['PCA (page)', 't-SNE (page)', 'UMAP 15 neighbours', 'UMAP 30 neighbours', 'UMAP 15, spread out']
for part in ('fin', 'fluke', 'body', 'all'):
    sel = [r for r in rows if (part == 'all' or r['part'] == part)]; k = len({(r['day'], r['part']) for r in sel})
    print(f"\n{part}: {k} day-part clouds, {sum(r['n'] for r in sel if r['name'] == names[0])} points; ceiling for 'animals' (in the 512 numbers) {np.nanmean([r['ceil'] for r in sel if r['name'] == names[0]]):.2f}")
    for nm in names:
        r = [x for x in sel if x['name'] == nm]; print(f"   {nm:22s} neighbours kept {np.mean([x['nb'] for x in r]):.2f} | animals {np.nanmean([x['pur'] for x in r]):.2f} | far apart {np.mean([x['rho'] for x in r]):.2f}")
s = [r for r in rows if r['name'] == '_stab']; print(f"\nanother random start, share of a point's 10 picture-neighbours that stay: t-SNE {np.mean([r['tsne'] for r in s]):.2f}, UMAP {np.mean([r['umap'] for r in s]):.2f}")
print(f"time for all {len(s)} clouds: t-SNE {tm['tsne']:.0f} s, UMAP (three settings) {tm['umap']:.0f} s")
