"""Pack one real Krea 2 capture (rs_kvshare RSCapture + the lead/lag tracking of the same picture) for Godot.

Run with the ComfyUI venv:  /mnt/nvme1tb/ComfyUI/venv/bin/python tools/export_data.py
Binary files are raw little-endian arrays; meta.json lists every file with its shape and dtype."""
import json, os, sys
import numpy as np
from PIL import Image

EXP = "/mnt/nvme1tb/ComfyUI/custom_nodes/rs_kvshare/experiments"
CAP = f"{EXP}/studio_capture"
IMG = "/mnt/nvme1tb/ComfyUI/output/studio"
NAME = "head_s23"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
sys.path.insert(0, EXP)
import kvtest, leadlag_gen as LG                                         # noqa: E402

H, W, S, B, P, NH = 48, 84, 8, 28, 10, 48
meta = json.load(open(f"{CAP}/meta.json"))
T = meta["steps"][0]["n_txt"]
files = {}


def save(name, arr, dtype):
    a = np.ascontiguousarray(arr, dtype=dtype)
    a.tofile(f"{OUT}/{name}")
    files[name] = {"shape": list(a.shape), "dtype": str(a.dtype)}


def u8_pos(x, axis_keep):
    """0..1 by the 99th percentile over the non-kept axes -> u8"""
    red = tuple(i for i in range(x.ndim) if i not in axis_keep)
    hi = np.percentile(np.abs(x), 99, axis=red, keepdims=True) + 1e-9
    return np.clip(x / hi * 255, 0, 255).astype(np.uint8), hi.squeeze()


def u8_stretch(x, axis_keep):
    """p1..p99 over the non-kept axes -> 0..255 (contrast per map)"""
    red = tuple(i for i in range(x.ndim) if i not in axis_keep)
    lo = np.percentile(x, 1, axis=red, keepdims=True)
    hi = np.percentile(x, 99, axis=red, keepdims=True)
    return np.clip((x - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def u8_signed(x, axis_keep):
    red = tuple(i for i in range(x.ndim) if i not in axis_keep)
    hi = np.percentile(np.abs(x), 99.5, axis=red, keepdims=True) + 1e-9
    return np.clip(128 + x / hi * 127, 0, 255).astype(np.uint8), hi.squeeze()


steps = [np.load(f"{CAP}/step_{s}.npz") for s in range(S)]
G = lambda k: np.stack([np.stack([steps[s][f"b{b}_{k}"].astype(np.float32) for b in range(B)]) for s in range(S)])

# ---------- images: the noisy input and the model's guess at every step
for kind in ("xt", "x0"):
    for s in range(S):
        Image.open(f"{IMG}/{kind}_{s + 1:05d}_.png").convert("RGB").resize((672, 384), Image.LANCZOS) \
            .save(f"{OUT}/img/{kind}_{s}.jpg", quality=88)
Image.open(f"{IMG}/x0_00008_.png").convert("RGB").save(f"{OUT}/img/final.jpg", quality=92)

# ---------- residual stream
pca = G("pca")                                                          # (S, B, 4032, 3)
a, _ = u8_signed(pca, (0, 1, 3)); save("pca.bin", a, np.uint8)
for k in ("norm", "write_attn", "write_mlp"):
    x = G(k)
    save(f"{k}.bin", u8_stretch(x, (0, 1)), np.uint8)
    meta[f"{k}_mean"] = x.mean(2).round(3).tolist()
# Ashwin's lock-and-key idea: does the MLP write where attention wrote (key fits lock), and is it more focused (lens)?
wa, wm = G("write_attn"), G("write_mlp")
def focus(x):                                   # share of the total write landing on the top 10 % of patches
    k = x.shape[-1] // 10
    return np.sort(x, -1)[..., -k:].sum(-1) / x.sum(-1)
meta["keylock_r"] = [[float(np.corrcoef(wa[s_, b_], wm[s_, b_])[0, 1]) for b_ in range(B)] for s_ in range(S)]
meta["focus_attn"] = focus(wa).round(3).tolist()
meta["focus_mlp"] = focus(wm).round(3).tolist()
meta["norm_txt"] = G("norm_txt").round(1).tolist()
meta["write_attn_txt_mean"] = G("write_attn_txt").mean(2).round(3).tolist()
meta["write_mlp_txt_mean"] = G("write_mlp_txt").mean(2).round(3).tolist()
meta["embed_norm_mean"] = [float(steps[s]["embed_norm"].astype(np.float32).mean()) for s in range(S)]
meta["txt_norm"] = [steps[s]["txt_norm"].round(2).tolist() for s in range(S)]

# ---------- attention: every head, every probe
ai = G("att_probe_img")                                                 # (S, B, P, 48, 12, 21)
mx = ai.max(axis=(4, 5), keepdims=True) + 1e-9
save("att_img.bin", np.clip(ai / mx * 255, 0, 255), np.uint8)
save("att_img_max.bin", mx.squeeze(), np.float32)
at = G("att_probe_txt")                                                 # (S, B, P, 48, T)
mt = at.max(axis=4, keepdims=True) + 1e-9
save("att_txt.bin", np.clip(at / mt * 255, 0, 255), np.uint8)
save("att_txt_max.bin", mt.squeeze(), np.float32)
save("att_txt_sum.bin", at.sum(4), np.float32)
for k in ("head_text_share", "head_dist", "head_entropy", "gate_img", "gate_txt", "mod"):
    meta[k] = G(k).round(4).tolist()
save("gate_probe.bin", np.clip(G("gate_probe") * 255, 0, 255), np.uint8)   # (S, B, P, 48)

# ---------- MLP neurons
nm = G("neuron_mean")                                                   # (S, B, 16384)
a, sc = u8_signed(nm, (0, 1)); save("neuron_mean.bin", a, np.uint8); meta["neuron_mean_scale"] = sc.round(4).tolist()
mp = G("neuron_maps")                                                   # (S, B, 16, 4032)
a, sc = u8_signed(mp, (0, 1, 2)); save("neuron_maps.bin", a, np.uint8); meta["neuron_maps_scale"] = sc.round(4).tolist()
meta["neuron_top"] = np.stack([np.stack([steps[s][f"b{b}_neuron_top"] for b in range(B)]) for s in range(S)]).tolist()
meta["probe_neurons"] = np.stack([np.stack([steps[s][f"b{b}_probe_neurons"] for b in range(B)]) for s in range(S)]).tolist()
meta["probe_neuron_vals"] = G("probe_neuron_vals").round(3).tolist()

# ---------- real weights (dequantized base model)
wz = np.load(f"{CAP}/weights.npz")
MATS = ["wq", "wk", "wv", "gate", "wo", "mlp_gate", "mlp_up", "mlp_down"]
wt = np.stack([np.stack([wz[f"{b}_{m}"].astype(np.float32) for m in MATS]) for b in range(B)])   # (B, 8, 64, 64)
save("weights.bin", u8_stretch(wt, (0, 1)), np.uint8)
meta["weight_mats"] = MATS
meta["weight_mean"] = wt.mean((2, 3)).round(5).tolist()
for m in ("wq", "wk", "wv", "wo"):
    meta[f"headnorm_{m}"] = [wz[f"{b}_{m}_headnorm"].round(3).tolist() for b in range(B)]
no = np.stack([wz[f"{b}_neuron_out_norm"].astype(np.float32) for b in range(B)])
a, _ = u8_pos(no, (0,)); save("neuron_out_norm.bin", a, np.uint8)

# ---------- text side
action = f"{LG.PRE} {LG.HEIGHTS['head']}"
prompt = f"{kvtest.CHAR}, {action}, {kvtest.SCENE}, {kvtest.STYLE}"
ids = LG.TOK(prompt)["input_ids"]
toks = [LG.TOK.decode([i]) for i in ids] + ["<end>"] * (T - len(ids))
meta["prompt"] = prompt
meta["tokens"] = toks[:T]
rz = np.load(f"{EXP}/leadlag/regions_head.npz")
meta["groups"] = {k[4:]: [int(rz[k][0]), int(rz[k][-1])] for k in rz.files}
meta["textenc_layer_norm"] = np.load(f"{CAP}/textenc_layer_norm.npy").round(2).tolist()

# ---------- tracking: SDPose on every step's guess, word maps, body regions, background
kp = json.load(open(f"{EXP}/leadlag/{NAME}_kp.json"))
track = []
for fr in kp:
    if not fr["people"]:
        track.append(None); continue
    pp = max(fr["people"], key=lambda p: sum(p["pose_keypoints_2d"][2::3]))
    track.append({k: [round(v, 2) for v in pp[k]] for k in ("pose_keypoints_2d", "hand_right_keypoints_2d",
                                                             "hand_left_keypoints_2d", "face_keypoints_2d")})
meta["track"] = track
A = np.load(f"{EXP}/leadlag/{NAME}_att.npz")
groups = list(A["groups"]); txt = A["txt"].astype(np.float32)          # (S, 4032, G+1)
WG = ["char", "holding", "book", "hands", "where"]
z = np.zeros((S, len(WG), H * W), np.float32)
for s in range(S):
    for gi, g in enumerate(WG):
        sh = txt[s, :, groups.index(g)] / np.maximum(txt[s, :, groups.index("all_text")], 1e-9)
        z[s, gi] = (sh - sh.mean()) / (sh.std() + 1e-9)
save("words.bin", np.clip((z + 1) / 8 * 255, 0, 255), np.uint8)      # z in [-1, 7]
meta["word_groups"] = WG
cent = {}
for gi, g in enumerate(("book", "hands")):
    cent[g] = []
    for s in range(S):
        ys, xs = np.nonzero(z[s, WG.index(g)].reshape(H, W) > 3)
        cent[g].append([float(ys.mean()), float(xs.mean())] if len(ys) else None)
meta["word_centroid"] = cent

sys.argv = [sys.argv[0]]
import mech  # noqa: E402  (reuses its body/zmap/segment helpers; prints its own report)
b, hands = mech.body(kp[-1])
book = mech.zmap(txt[-1], A["groups"], "book") > 3
handm = np.zeros((H, W), bool)
for hk in hands:
    for x, y, sc_ in hk:
        if sc_ > 0.3: handm |= mech.disk(y / 16, x / 16, 1)
fore = mech.segment(b[4], b[3], 1.2) | mech.segment(b[7], b[6], 1.2)
upper = mech.segment(b[3], b[2], 1.2) | mech.segment(b[6], b[5], 1.2)
shoul = mech.disk(b[2, 1] / 16, b[2, 0] / 16, 1.5) | mech.disk(b[5, 1] / 16, b[5, 0] / 16, 1.5)
face = mech.disk(b[0, 1] / 16, b[0, 0] / 16, 3)
charm = mech.zmap(txt[-1], A["groups"], "char") > 0.5
REG = ["background", "torso", "face", "shoulder", "upper arm", "forearm", "hands", "book"]
lab = np.zeros((H, W), np.uint8)
for i, m in enumerate([None, charm, face, shoul, upper, fore, handm, book]):
    if m is not None: lab[m] = i
save("labels.bin", lab, np.uint8)
meta["regions"] = REG

# patch -> patch attention (mean over heads, blocks 4/10/16/22), top 32 cells of 2x2 patches, for ANY patch
img = A["img"].astype(np.float32)                                       # (S, 4032, 1008)
top = np.argsort(-img, axis=2)[:, :, :32]
val = np.take_along_axis(img, top, 2)
save("patchatt_idx.bin", top, np.uint16)
save("patchatt_val.bin", np.clip(val / (val[:, :, :1] + 1e-9) * 255, 0, 255), np.uint8)
save("patchatt_mass.bin", val.sum(2), np.float32)

meta["files"] = files
meta["shape"] = {"S": S, "B": B, "P": P, "heads": NH, "kv_heads": 12, "T": T, "H": H, "W": W, "mlp": 16384,
                 "width": 6144, "att_cells": [12, 21], "patch_cells": [24, 42]}
json.dump(meta, open(f"{OUT}/meta.json", "w"))
print("exported", sum(os.path.getsize(f"{OUT}/{f}") for f in files) / 1e6, "MB binary;", len(meta["tokens"]), "tokens")
print("tokens sample:", meta["tokens"][:6], meta["tokens"][-7:], meta["groups"])
