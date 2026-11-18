"""Fine-tune HW-b6 on the first N pool days of bench.splits()[0] (dorsal instances with a catalogue animal).

    flock LOCK env OMP_NUM_THREADS=8 .venv/bin/python experiments/moonshot_sizing/fin_tune/train.py --name s5_0 --days 5 --seed 0

Classes = day | catalogue animal. Each batch = P animals x K crops from ONE day: an anchor animal seen in >= 2 safe
clusters, the other animals of its catalogue group (hard negatives), filled up with other animals of the day; the K crops
of an animal are spread round-robin over its safe clusters (different surfacings).
Loss = sub-centre ArcFace (k 2, s 30, m 0.3; centres start at the class means of the raw embedding, so the head does not
pull the backbone around at the start) + supervised contrastive (T 0.1) whose positives are the same animal in a
DIFFERENT safe cluster; same-cluster crops of the anchor's animal are left out of numerator and denominator (near-
duplicate frames). No horizontal flip (a flip turns a left flank into a mirrored left flank, not a right flank).
Resumable: CACHE/runs/<name>/resume.pt every --ckpt-every steps and when --max-minutes is used up (exit code 3).
Per-step randomness comes from (seed, step), so a resumed run draws the same batches.
"""
import argparse
import collections
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

from common import CACHE, CTX, MEAN, RES, TRAIN_PX, group_of, hw_raw, split, unit
from model import HW


def load_index(days):
    items, by = [], collections.defaultdict(lambda: collections.defaultdict(list))   # by[(day, animal)][cluster] -> item ids
    for d in days:
        for n, key, c, a in json.load(open(CACHE / 'crops' / d / 'index.json')):
            by[(d, a)][c].append(len(items))
            items.append((d, n, key, c, a))
    return items, by


class Crops(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, t):
        i, s = t
        d, n = self.items[i][:2]
        r = np.random.default_rng(s)
        im = Image.open(CACHE / 'crops' / d / f'{n}.jpg').convert('RGB')
        if r.random() < 0.5:
            im = im.rotate(r.uniform(-8, 8), Image.BICUBIC)
        unit_px = TRAIN_PX / CTX                                 # the box's longer side, in crop pixels
        side = r.uniform(1.0, 1.3) * unit_px
        cx = TRAIN_PX / 2 + r.uniform(-0.05, 0.05) * unit_px
        cy = TRAIN_PX / 2 + r.uniform(-0.05, 0.05) * unit_px
        im = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).resize((RES, RES), Image.BILINEAR)
        for E in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            im = E(im).enhance(r.uniform(0.75, 1.25))
        if r.random() < 0.2:
            im = im.filter(ImageFilter.GaussianBlur(r.uniform(0.5, 1.5)))
        return torch.from_numpy(np.asarray(im, np.uint8).copy()).permute(2, 0, 1), i


class Batches(torch.utils.data.Sampler):
    def __init__(self, items, by, P, K, seed, start, steps):
        self.P, self.K, self.seed, self.start, self.steps = P, K, seed, start, steps
        self.by = by
        self.days = sorted({d for d, a in by})
        self.multi = {d: [a for (dd, a), cl in by.items() if dd == d and len(cl) >= 2] for d in self.days}
        self.animals = {d: [a for (dd, a) in by if dd == d] for d in self.days}
        self.days = [d for d in self.days if self.multi[d]]
        w = np.array([len(self.multi[d]) for d in self.days], float)
        self.w = w / w.sum()

    def __len__(self):
        return self.steps - self.start

    def batch(self, step):
        r = np.random.default_rng([self.seed, step])
        d = self.days[r.choice(len(self.days), p=self.w)]
        anchor = self.multi[d][r.integers(len(self.multi[d]))]
        g = group_of(anchor)
        same = [a for a in self.animals[d] if a != anchor and group_of(a) == g]
        rest = [a for a in self.animals[d] if a != anchor and group_of(a) != g]
        r.shuffle(same), r.shuffle(rest)
        same.sort(key=lambda a: -min(len(self.by[(d, a)]), 2))          # prefer group mates seen in >= 2 clusters
        chosen = ([anchor] + same + rest)[:self.P]
        out = []
        for a in chosen:
            cl = list(self.by[(d, a)].values())
            r.shuffle(cl)
            for j in range(self.K):
                ids = cl[j % len(cl)]
                out.append((int(ids[r.integers(len(ids))]), int(r.integers(2**31))))
        return out

    def __iter__(self):
        for s in range(self.start, self.steps):
            yield self.batch(s)


def supcon(z, animal, cluster, T=0.1):
    sim = z @ z.T / T
    sa = animal[:, None] == animal[None, :]
    sc = cluster[:, None] == cluster[None, :]
    pos = sa & ~sc
    excl = sa & sc                                                          # self and same-cluster frames of the animal
    sim = sim.masked_fill(excl, float('-inf'))
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0, 0
    l = -(logp.masked_fill(~pos, 0).sum(1)[has] / pos.sum(1)[has])
    return l.mean(), int(has.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--days', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--P', type=int, default=4)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--unfreeze-from', type=int, default=5)
    ap.add_argument('--lr-bb', type=float, default=2e-5)
    ap.add_argument('--lr-emb', type=float, default=1e-4)
    ap.add_argument('--lr-arc', type=float, default=1e-3)
    ap.add_argument('--w-arc', type=float, default=1.0)
    ap.add_argument('--w-sc', type=float, default=1.0)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--ckpt-every', type=int, default=100)
    ap.add_argument('--max-minutes', type=float, default=24.0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--mem-gb', type=float, default=10.0)
    ap.add_argument('--cpu', action='store_true', help='smoke test only')
    a = ap.parse_args()
    t0 = time.time()
    hold, pool, test = split()
    days = pool[:a.days]
    assert not set(days) & set(test)
    out = CACHE / 'runs' / a.name
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cpu' if a.cpu else 'cuda')
    if not a.cpu:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(a.seed)

    items, by = load_index(days)
    classes = sorted(by)
    cid = {k: i for i, k in enumerate(classes)}
    lab = torch.tensor([cid[(d, an)] for d, n, key, c, an in items], device=dev)
    clu = torch.tensor([hash((d, c)) % (2**62) for d, n, key, c, an in items], device=dev)

    model = HW(a.unfreeze_from).to(dev).eval()
    # sub-centre ArcFace weights from the raw embedding's class means (k = 2 sub-centres, tiny noise on the second)
    raw = {d: hw_raw(d) for d in days}
    W = np.stack([unit(np.stack([raw[d][items[i][2]] for cl in by[k].values() for i in cl]).mean(0, keepdims=True))[0] for k in classes for d in [k[0]]])
    g = torch.Generator().manual_seed(a.seed)
    W = torch.tensor(W).repeat_interleave(2, 0)
    W = W + 0.01 * torch.randn(W.shape, generator=g) * torch.tensor([0., 1.]).repeat(len(classes))[:, None]
    arc = torch.nn.Parameter(W.to(dev))
    emb_p = list(model.net.bn.parameters()) + list(model.net.pool.parameters())
    bb_p = [p for p in model.back.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=bb_p, lr=a.lr_bb, weight_decay=1e-4), dict(params=emb_p, lr=a.lr_emb, weight_decay=0.),
                             dict(params=[arc], lr=a.lr_arc, weight_decay=0.)])
    base_lr = [gr['lr'] for gr in opt.param_groups]
    start = 0
    rp = out / 'resume.pt'
    if rp.exists():
        ck = torch.load(rp, map_location=dev, weights_only=True)
        model.load_state_dict(ck['model']); arc.data.copy_(ck['arc']); opt.load_state_dict(ck['opt']); start = ck['step']
        print(f'resumed at step {start}', flush=True)
    n_tr = sum(p.numel() for p in model.trainable())
    print(f'{a.name}: {len(days)} days {days}, {len(items)} crops, {len(classes)} classes, trainable {n_tr / 1e6:.1f} M of '
          f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, steps {start} -> {a.steps}', flush=True)

    def save(step, path=rp):
        torch.save(dict(model=model.state_dict(), arc=arc.data, opt=opt.state_dict(), step=step, args=vars(a)), str(path) + '.tmp')
        (out / (path.name + '.tmp')).replace(path)

    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
    sampler = Batches(items, by, a.P, a.K, a.seed, start, a.steps)
    dl = torch.utils.data.DataLoader(Crops(items), batch_sampler=sampler, num_workers=a.workers, persistent_workers=False, prefetch_factor=2)
    log = collections.defaultdict(list)
    step, tl = start, time.time()
    for x, idx in dl:
        x = (x.to(dev, non_blocking=True).float() / 255. - mean) / std
        idx = idx.to(dev)
        for gr, lr in zip(opt.param_groups, base_lr):
            f = (step + 1) / a.warmup if step < a.warmup else 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))
            gr['lr'] = lr * f
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            e = model(x)
        e = F.normalize(e.float(), dim=1)
        y, c = lab[idx], clu[idx]
        cos = (e @ F.normalize(arc, dim=1).T).view(len(e), -1, 2).amax(2)
        tgt = cos.gather(1, y[:, None]).clamp(-1 + 1e-6, 1 - 1e-6)
        cos_m = cos.scatter(1, y[:, None], torch.cos(torch.acos(tgt) + 0.3))
        l_arc = F.cross_entropy(30 * cos_m, y)
        l_sc, n_anchor = supcon(e, y, c)
        loss = a.w_arc * l_arc + a.w_sc * l_sc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable() + [arc], 5.0)
        opt.step()
        step += 1
        with torch.no_grad():
            S = e @ e.T
            sa, sc = y[:, None] == y[None, :], c[:, None] == c[None, :]
            log['arc'].append(l_arc.item()); log['sc'].append(float(l_sc)); log['anch'].append(n_anchor)
            log['pos_x'].append(S[sa & ~sc].mean().item() if (sa & ~sc).any() else np.nan)
            log['neg'].append(S[~sa].mean().item())
        if step % 25 == 0 or step == a.steps:
            print(f'step {step:5d}  arc {np.mean(log["arc"]):.3f}  supcon {np.mean(log["sc"]):.3f}  anchors {np.mean(log["anch"]):.1f}  '
                  f'cos same animal other cluster {np.nanmean(log["pos_x"]):.3f}  other animals {np.mean(log["neg"]):.3f}  '
                  f'{(time.time() - tl) / 25:.2f} s/step  peak GPU {torch.cuda.max_memory_allocated() / 2**30 if not a.cpu else 0:.2f} GB', flush=True)
            log.clear(); tl = time.time()
        if step == a.steps:
            break
        if step % a.ckpt_every == 0:
            save(step)
        if time.time() - t0 > a.max_minutes * 60:
            save(step)
            print(f'time limit at step {step}, saved, {time.time() - t0:.0f} s', flush=True)
            raise SystemExit(3)
    torch.save(model.net.state_dict(), str(out / 'final.pt'))
    rp.unlink(missing_ok=True)
    print(f'DONE {a.name} step {step}, {time.time() - t0:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30 if not a.cpu else 0:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
