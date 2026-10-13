"""Canonical-surface model: for every pixel, *where on the whale* it is, as a 3D point on the
undeformed mesh, plus a rigid pose per crop.

Dense head (per pixel): the folded, normalised canonical point (x_hat = body fraction tail ->
snout, y_hat = |y| / L lateral distance from the midline, z_hat = z / L height), and a
visibility logit. Pose head (per crop): 6D rotation -> camera-from-whale, log pixels-per-metre.

Why folded: the body is bilaterally symmetric, so which flank a patch of skin is on cannot be
read locally; it follows from the whole whale's orientation. Keeping the sign out of the dense
target means a mirrored crop has an identical target, the network never has to guess it
pixel by pixel, and the side is decided once per whale afterwards (geometry.resolve_side) or
at matching time. Why 3D points instead of (u, angle): the angle around the body axis is
degenerate on the flukes and fins (a whole fluke blade spans ~18 degrees), 3D points are not.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.backbone import DinoFeatures, normalize_image
from ..common.data import IGNORE
from ..common.geometry import rotation_from_6d_torch
from ..common.heads import DenseDecoder
from ..common.run03_labels import DEFORM_PARAMS, KEYPOINTS

HEATMAP_STRIDE = 4            # keypoint heatmaps are predicted on the decoder's stride-4 grid
KP_STATES = ["visible", "under_water", "hidden", "out_of_frame"]   # the head's classes (factory hidden + occluded -> hidden)
DEFORM_SCALE_DEG = 30.0       # deformation parameters are regressed in units of this many degrees


class WhaleCanonModel(nn.Module):
    def __init__(self, arch: str = "vitb16", unfreeze_blocks: int = 4, layers=None, weights=None, repo=None, pretrained: bool = True,
                 decoder_width: int = 256, keypoints: int = 0):
        super().__init__()
        self.config = {"arch": arch, "unfreeze_blocks": unfreeze_blocks, "layers": layers, "decoder_width": decoder_width, "task": "canon",
                       "keypoints": keypoints}
        self.backbone = DinoFeatures(arch, weights=weights, repo=repo, layers=layers, pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        self.decoder = DenseDecoder(self.backbone.embed_dim, len(self.backbone.layers), 4, width=decoder_width)
        self.pool_logit = nn.Conv2d(decoder_width, 1, 1)
        self.pose = nn.Sequential(
            nn.Linear(decoder_width + self.backbone.embed_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 7),
        )
        with torch.no_grad():
            self.pose[-1].weight.mul_(0.1)
            self.pose[-1].bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 3.5], dtype=torch.float32))
        # run03 heads (optional): keypoint heatmaps on the stride-4 features, a visibility state per
        # keypoint and the body deformation parameters from the pooled crop features.
        self.keypoints = int(keypoints)
        if self.keypoints:
            fine_width = self.decoder.out.in_channels
            self.kp_head = nn.Sequential(nn.Conv2d(fine_width, fine_width, 3, padding=1), nn.GELU(), nn.Conv2d(fine_width, self.keypoints, 1))
            self.aux = nn.Sequential(nn.Linear(decoder_width + self.backbone.embed_dim, 512), nn.GELU(), nn.Dropout(0.1))
            self.kp_state = nn.Linear(512, self.keypoints * len(KP_STATES))
            self.deform = nn.Linear(512, len(DEFORM_PARAMS))
            with torch.no_grad():
                self.deform.weight.mul_(0.1); self.deform.bias.zero_()
                self.kp_head[-1].bias.fill_(-4.0)        # heatmaps start near zero (sigmoid(-4)), as the targets mostly are

    def forward(self, image01: torch.Tensor) -> dict:
        x = normalize_image(image01)
        feats, cls = self.backbone(x)
        dense, coarse, fine = self.decoder.forward_full(feats, x)
        xyz = torch.cat([torch.sigmoid(dense[:, 0:1]), dense[:, 1:3]], 1)     # x_hat in [0, 1]; y_hat, z_hat free
        vis_logit = dense[:, 3]
        att = torch.softmax(self.pool_logit(coarse).flatten(1), dim=1)
        pooled = (coarse.flatten(2) * att[:, None, :]).sum(-1)
        feat = torch.cat([pooled, cls], 1)
        pose = self.pose(feat)
        R = rotation_from_6d_torch(pose[:, :6])
        out = {"xyz": xyz, "vis_logit": vis_logit, "R": R, "d6": pose[:, :6], "log_ppm": pose[:, 6]}
        if self.keypoints:
            aux = self.aux(feat)
            out["kp_heat"] = torch.sigmoid(self.kp_head(fine))                                     # (B, K, H/4, W/4)
            out["kp_state_logit"] = self.kp_state(aux).view(-1, self.keypoints, len(KP_STATES))   # (B, K, 4)
            out["deform"] = self.deform(aux)                                                       # (B, 7), units of DEFORM_SCALE_DEG
        return out

    def param_groups(self, lr_head: float, lr_backbone: float, layer_decay: float, weight_decay: float) -> list[dict]:
        groups = self.backbone.param_groups(lr_backbone, layer_decay, weight_decay)
        decay, no_decay = [], []
        modules = [self.decoder, self.pool_logit, self.pose] + ([self.kp_head, self.aux, self.kp_state, self.deform] if self.keypoints else [])
        for module in modules:
            for name, p in module.named_parameters():
                (no_decay if p.ndim <= 1 else decay).append(p)
        groups.append({"params": decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head_wd"})
        groups.append({"params": no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "head_nd"})
        for g in groups:
            g["base_lr"] = g["lr"]
        return groups

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from ..common.train_utils import load_checkpoint

        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]["model"]
        model = cls(arch=cfg["arch"], unfreeze_blocks=0, layers=cfg.get("layers"), decoder_width=cfg.get("decoder_width", 256), pretrained=False,
                    keypoints=int(cfg.get("keypoints", 0)))
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


def canon_loss(out: dict, batch: dict, weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    """Visibility BCE (ignore class excluded), L1 on the folded coordinates over valid pixels,
    Frobenius on the rotation, L1 on log scale."""
    w = {"vis": 1.0, "xyz": 10.0, "rot": 2.0, "scale": 1.0}
    w.update(weights or {})
    mask = batch["mask"]
    not_ignored = mask != IGNORE
    vis_t = (mask == 1).float()
    vis = F.binary_cross_entropy_with_logits(out["vis_logit"], vis_t, reduction="none")
    vis = (vis * not_ignored).sum() / not_ignored.sum().clamp(min=1)
    valid = batch["valid"]
    n = valid.sum().clamp(min=1)
    l1 = (out["xyz"] - batch["xyz"]).abs().mean(1)          # (B, H, W): mean over the 3 coordinates
    xyz = (l1 * valid).sum() / n
    rot = ((out["R"] - batch["R"]) ** 2).sum(dim=(1, 2)).mean()
    scale = (out["log_ppm"] - batch["log_ppm"]).abs().mean()
    loss = w["vis"] * vis + w["xyz"] * xyz + w["rot"] * rot + w["scale"] * scale
    parts = {"vis": vis.item(), "xyz_l1": xyz.item(), "rot": rot.item(), "scale": scale.item()}
    if "kp_heat" in out and "kp_heat" in batch:
        wk = {"kp": 100.0, "kp_state": 0.5, "deform": 1.0}
        wk.update({k: v for k, v in (weights or {}).items() if k in wk})
        # heatmaps: MSE against the Gaussian targets (zero maps for keypoints that are not visible)
        heat = ((out["kp_heat"] - batch["kp_heat"]) ** 2).mean()
        # state: cross-entropy per keypoint, -1 = missing from the shot
        st = batch["kp_state"]
        known = st >= 0
        state = F.cross_entropy(out["kp_state_logit"][known], st[known]) if known.any() else out["kp_state_logit"].sum() * 0
        deform = (out["deform"] - batch["deform"]).abs().mean()
        loss = loss + wk["kp"] * heat + wk["kp_state"] * state + wk["deform"] * deform
        parts.update({"kp_heat": heat.item(), "kp_state": state.item(), "deform": deform.item()})
    return loss, parts


def decode_heatmaps(heat: "np.ndarray", stride: int = HEATMAP_STRIDE) -> tuple["np.ndarray", "np.ndarray"]:
    """(K, h, w) heatmaps -> keypoint positions (K, 2) in crop pixels (peak, refined by the local
    centre of mass of a 5x5 window) and peak confidences (K,)."""
    import numpy as np

    K, h, w = heat.shape
    flat = heat.reshape(K, -1)
    idx = flat.argmax(1)
    conf = flat[np.arange(K), idx]
    py, px = np.divmod(idx, w)
    uv = np.zeros((K, 2), np.float32)
    for k in range(K):
        y0, y1 = max(py[k] - 2, 0), min(py[k] + 3, h)
        x0, x1 = max(px[k] - 2, 0), min(px[k] + 3, w)
        win = heat[k, y0:y1, x0:x1].astype(np.float64)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        wsum = win.sum()
        cy, cx = ((ys * win).sum() / wsum, (xs * win).sum() / wsum) if wsum > 1e-9 else (py[k], px[k])
        uv[k] = ((cx + 0.5) * stride, (cy + 0.5) * stride)
    return uv, conf.astype(np.float32)


def consistency_loss(out_a: dict, out_b: dict, whale: torch.Tensor, w_rot: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Label-free mirror consistency: out_a is the model on a crop, out_b on its mirror image.
    The folded coordinates of corresponding pixels must agree (L1 over the whale region, both
    branches trained), and the rotation must be the mirrored rotation. Used on real photos
    alongside the supervised synthetic loss, which anchors the coordinates to the mesh."""
    from ..common.geometry import flip_rotation_torch

    xyz_b = out_b["xyz"].flip(-1)
    region = whale.float()
    n = region.sum().clamp(min=1)
    coord = ((out_a["xyz"] - xyz_b).abs().mean(1) * region).sum() / n
    R_b = flip_rotation_torch(out_b["R"])
    rot = ((out_a["R"] - R_b) ** 2).sum(dim=(1, 2)).mean()
    loss = coord + w_rot * rot
    return loss, {"cons_xyz": coord.item(), "cons_rot": rot.item()}

