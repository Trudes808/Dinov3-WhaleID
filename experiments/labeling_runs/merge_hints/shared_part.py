"""When a stack comes up in step 2 and its whale is already an animal: does any stack sorted to that animal so far show
the same part (a fluke with a fluke, a dorsal fin with a dorsal fin)? If not, the right answer is "same whale" with
nothing comparable on the screen. Catalogue as the researcher (always right), per catalogue encounter; the page's order
(flukes in capture order, then dorsal fins, then the rest) against plain capture order."""
import os, collections
from sim import Day, SPLITS, BIT
alld = sorted(f[:-5] for f in os.listdir('cache') if f.endswith('.json')); FF = BIT['fluke'] | BIT['fin']
def walk(day, order):
    c = collections.Counter(); hit = set(); animals = set()
    for g, cs in day.scope.items():
        stacks = [i for a in cs for i in day.animals[a]]
        if order == 'page': stacks.sort(key=lambda i: (0 if day.bits[i] & BIT['fluke'] else 1 if day.bits[i] & BIT['fin'] else 2, day.t0[i], i))
        else: stacks.sort(key=lambda i: (day.t0[i], i))
        seen = collections.defaultdict(list)
        for s in stacks:
            a = day.cat[s]; animals.add((g, a)); b = day.bits[s]
            if not seen[a]: c['first stack of its whale'] += 1
            else:
                phase = 'fluke' if b & BIT['fluke'] else 'fin' if b & BIT['fin'] else 'rest'
                if any(b & day.bits[t] & FF for t in seen[a]):
                    side = b & (BIT['finL'] | BIT['finR']); fl = any(b & day.bits[t] & BIT['fluke'] for t in seen[a])
                    if not fl and side and not any(day.bits[t] & side for t in seen[a]) and any(day.bits[t] & (BIT['finL'] | BIT['finR']) for t in seen[a]): c['a part in common, but only the other fin side'] += 1
                    else: c['a part in common'] += 1
                elif phase == 'rest': c['shows neither fluke nor fin'] += 1
                else: c['NO part in common (' + phase + ' stack)'] += 1; hit.add((g, a))
            seen[a].append(s)
    return c, len(hit), len(animals)
for tag, days in (('val', SPLITS['val']), ('eval', SPLITS['eval']), ('train', [d for d in alld if d not in SPLITS['val'] + SPLITS['eval']]), ('2024-12-23', ['2024-12-23'])):
    for order in ('page', 'capture'):
        tot = collections.Counter(); nh = na = 0
        for d in days:
            c, h, a = walk(Day(d), order); tot.update(c); nh += h; na += a
        q = sum(v for k, v in tot.items() if k != 'first stack of its whale'); no = sum(v for k, v in tot.items() if k.startswith('NO part'))
        print(f"{tag:10s} {order:8s} order: {sum(tot.values())} stacks, {tot['first stack of its whale']} first of their whale; of the other {q}: part in common {tot['a part in common']}, only the other fin side {tot['a part in common, but only the other fin side']}, "
              f"no part in common {no} ({no / max(q, 1):.1%}; fluke stacks {tot['NO part in common (fluke stack)']}, fin stacks {tot['NO part in common (fin stack)']}), neither part {tot['shows neither fluke nor fin']} | whales with such a moment {nh} of {na}")
