"""What a "second look before a new animal" would cost and reach. Catalogue as the researcher, the page's order. For
every stack that comes up while its encounter already has animals: the page's similarity (blend) to the most similar
animal. A stack that truly starts a new whale with a look-alike above the bar = a second look for nothing; a stack of a
whale already there whose own animal scores below the bar = a miss the second look would not have caught."""
import os, collections, numpy as np
from sim import Day, SPLITS, BIT
alld = sorted(f[:-5] for f in os.listdir('cache') if f.endswith('.json'))
for tag, days in (('val', SPLITS['val']), ('eval', SPLITS['eval']), ('train', [d for d in alld if d not in SPLITS['val'] + SPLITS['eval']])):
    new_top, old_own, old_rank1 = [], [], []
    for d in days:
        day = Day(d)
        for g, cs in day.scope.items():
            stacks = sorted((i for a in cs for i in day.animals[a]), key=lambda i: (0 if day.bits[i] & BIT['fluke'] else 1 if day.bits[i] & BIT['fin'] else 2, day.t0[i], i)); seen = collections.defaultdict(list)
            for s in stacks:
                a = day.cat[s]
                if seen:
                    sc = {b: day.scores([s], t)['blend'] for b, t in seen.items() if t and not day.conflict([s], t)}
                    if sc:
                        if a in sc: old_own.append(sc[a]); old_rank1.append(max(sc, key=sc.get) == a)
                        else: new_top.append(max(sc.values()))
                seen[a].append(s)
    new_top, old_own = np.array(new_top), np.array(old_own)
    print(f"{tag}: stacks that start a new whale with other animals to compare {len(new_top)}; stacks of a whale already there {len(old_own)} (its own animal is the most similar {np.mean(old_rank1):.2f})")
    for t in (0.4, 0.5, 0.6, 0.7):
        print(f"   bar {t:.1f}: true new whales whose closest animal is above it {np.mean(new_top >= t):.2f} ({int((new_top >= t).sum())} second looks for nothing) | stacks whose own animal is above it {np.mean(old_own >= t):.2f} (a wrong 'new' there would get the second look), below it {int((old_own < t).sum())}")
