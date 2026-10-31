"""RSCapture: record one real Krea 2 generation end to end, for the Godot "Krea Studio".

Per denoising step (8) and block (28), from the real weights:
  - residual stream after every block: 3-component PCA of the image tokens (a colour picture of "what the tokens
    are" at that depth) + per-token norm, text tokens projected on the same basis
  - attention, all 48 heads: for each probe patch, its attention over the image (4x4-patch cells) and over every
    text token; per head over ALL queries: share of attention on text, mean attention distance, entropy
  - the sigmoid output gate per head (image mean, text mean, per probe)
  - how much attention and the MLP write into every token (|gate * update|)
  - SwiGLU MLP (16384 neurons): mean activation per neuron over the image, the 16 most spatially selective
    neurons with their maps, the top neurons at each probe
  - adaLN modulation magnitudes (pre/post scale, shift, gate)
Plus: x_t and the x0 guess per step (decode later), text-encoder layer norms, txtfusion layer mix, and per-block
weight thumbnails / per-head weight norms of the dequantized weights (base model, LoRA deltas not included).
Everything goes to out_dir as .npy / .json; export_godot.py packs it for Godot."""
import json, os
import numpy as np
import torch
import torch.nn.functional as F

H_P, W_P, CELL = 48, 84, 4


def _deq(w):
    if hasattr(w, "dequantize"):
        w = w.dequantize()
    return w.float()


def _thumb(w, size=64):
    a = w.abs().float()
    a = F.adaptive_avg_pool2d(a[None, None], (size, size))[0, 0]
    return a.cpu().numpy().astype(np.float16)


class RSCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "out_dir": ("STRING", {"default": "/tmp/rs_capture"}),
                             "probes": ("STRING", {"multiline": True, "default": "[]",
                                                   "tooltip": "JSON list of [name, row, col] patch coordinates"}),
                             "weights": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "rs/experiments"

    def patch(self, model, out_dir, probes, weights):
        os.makedirs(out_dir, exist_ok=True)
        m = model.clone()
        probes = json.loads(probes)
        pidx = torch.tensor([r * W_P + c for _, r, c in probes], dtype=torch.long)
        st = {"step": -1, "cur": None, "pca_prev": None, "dist": None, "weights_done": not weights}
        meta = {"probes": probes, "steps": [], "grid": [H_P, W_P], "cell": CELL}

        def block_store(i):
            return st["cur"]["blocks"].setdefault(i, {})

        def hooks_for(dm):
            hs = []
            n_txt = lambda: st["cur"]["n_txt"]

            def first_hook(mod, inp, out):
                x = out[0].float()
                st["cur"]["embed_norm"] = x.norm(dim=-1).cpu().numpy().astype(np.float16)
            hs.append(dm.first.register_forward_hook(first_hook))

            def txt_hook(mod, inp, out):
                st["cur"]["n_txt"] = out.shape[1]
                st["cur"]["txt_norm"] = out[0].float().norm(dim=-1).cpu().numpy()
            hs.append(dm.txtmlp.register_forward_hook(txt_hook))

            for i, blk in enumerate(dm.blocks):
                def mod_hook(mod, inp, out, i=i):
                    b = block_store(i)
                    b["_gates"] = (out[2], out[5])                  # pregate, postgate
                    b["mod"] = np.array([o.float().abs().mean().item() for o in out], np.float32)
                hs.append(blk.mod.register_forward_hook(mod_hook))

                def gate_hook(mod, inp, out, i=i):
                    b = block_store(i); T = n_txt()
                    g = torch.sigmoid(out[0].float()).reshape(out.shape[1], 48, -1).mean(-1)   # (L, 48)
                    b["gate_img"] = g[T:].mean(0).cpu().numpy()
                    b["gate_txt"] = g[:T].mean(0).cpu().numpy()
                    b["gate_probe"] = g[T + pidx.to(g.device)].cpu().numpy().astype(np.float16)
                hs.append(blk.attn.gate.register_forward_hook(gate_hook))

                def attn_out_hook(mod, inp, out, i=i):
                    b = block_store(i); T = n_txt()
                    w = (b["_gates"][0] * out).float()[0].norm(dim=-1)
                    b["write_attn"] = w[T:].cpu().numpy().astype(np.float16)
                    b["write_attn_txt"] = w[:T].cpu().numpy()
                hs.append(blk.attn.register_forward_hook(attn_out_hook))

                def mlp_out_hook(mod, inp, out, i=i):
                    b = block_store(i); T = n_txt()
                    w = (b["_gates"][1] * out).float()[0].norm(dim=-1)
                    b["write_mlp"] = w[T:].cpu().numpy().astype(np.float16)
                    b["write_mlp_txt"] = w[:T].cpu().numpy()
                hs.append(blk.mlp.register_forward_hook(mlp_out_hook))

                def neuron_hook(mod, inp, i=i):
                    b = block_store(i); T = n_txt()
                    h = inp[0][0].float()                            # (L, 16384)
                    hi = h[T:]
                    b["neuron_mean"] = hi.mean(0).cpu().numpy().astype(np.float16)
                    b["neuron_txt_mean"] = h[:T].mean(0).cpu().numpy().astype(np.float16)
                    sel = hi.std(0) / (hi.abs().mean(0) + 1e-3)      # spatially selective, not just loud
                    top = torch.topk(sel, 16).indices
                    b["neuron_top"] = top.cpu().numpy().astype(np.int32)
                    b["neuron_maps"] = hi[:, top].T.cpu().numpy().astype(np.float16)     # (16, 4032)
                    pv, pi = torch.topk(hi[pidx.to(h.device)], 8, dim=-1)
                    b["probe_neurons"] = pi.cpu().numpy().astype(np.int32)
                    b["probe_neuron_vals"] = pv.cpu().numpy().astype(np.float16)
                hs.append(blk.mlp.down.register_forward_pre_hook(neuron_hook))

                def block_hook(mod, inp, out, i=i):
                    b = block_store(i); T = n_txt()
                    x = out[0].float()
                    xi, xt = x[T:], x[:T]
                    mu = xi.mean(0, keepdim=True)
                    _, _, V = torch.pca_lowrank(xi - mu, q=3, center=False, niter=3)
                    pi_ = (xi - mu) @ V                              # (4032, 3)
                    prev = st["pca_prev"]
                    if prev is not None:                             # keep colours steady block to block
                        sgn = torch.sign((pi_ * prev).sum(0)); sgn[sgn == 0] = 1
                        pi_, V = pi_ * sgn, V * sgn
                    st["pca_prev"] = pi_
                    b["pca"] = pi_.cpu().numpy().astype(np.float16)
                    b["pca_txt"] = ((xt - mu) @ V).cpu().numpy().astype(np.float16)
                    b["norm"] = xi.norm(dim=-1).cpu().numpy().astype(np.float16)
                    b["norm_txt"] = xt.norm(dim=-1).cpu().numpy()
                    b.pop("_gates", None)
                hs.append(blk.register_forward_hook(block_hook))
            return hs

        def attn_patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
            from comfy.ldm.flux.math import apply_rope
            i = extra_options["block_index"]
            b = block_store(i)
            s0, s1 = extra_options["img_slice"]
            qr, kr = apply_rope(q, k, pe) if pe is not None else (q, k)
            Hq, rep = qr.shape[1], qr.shape[1] // kr.shape[1]
            dev = q.device
            if st["dist"] is None:
                yy, xx = torch.meshgrid(torch.arange(H_P, device=dev), torch.arange(W_P, device=dev), indexing="ij")
                pos = torch.stack([yy.flatten(), xx.flatten()], 1).float()
                st["dist"] = torch.cdist(pos, pos)                  # (4032, 4032) patches
            scale = qr.shape[-1] ** -0.5
            pq = pidx.to(dev) + s0
            pimg, ptxt, tshare, mdist, ent = [], [], [], [], []
            with torch.no_grad():
                for h in range(Hq):
                    kk = kr[0, h // rep].float()
                    p = torch.softmax(qr[0, h, s0:s1].float() @ kk.T * scale, -1)       # (4032, L)
                    tshare.append(p[:, :s0].sum(1).mean().item())
                    mdist.append(((p[:, s0:] * st["dist"]).sum(1) / p[:, s0:].sum(1).clamp_min(1e-9)).mean().item())
                    ent.append((-(p * (p + 1e-12).log()).sum(1)).mean().item())
                    pp = p[pq - s0]                                                       # (P, L)
                    im = pp[:, s0:].reshape(-1, 1, H_P, W_P)
                    pimg.append((F.avg_pool2d(im, CELL) * CELL * CELL)[:, 0])           # (P, 12, 21)
                    ptxt.append(pp[:, :s0])
            b["att_probe_img"] = torch.stack(pimg, 1).cpu().numpy().astype(np.float16)  # (P, 48, 12, 21)
            b["att_probe_txt"] = torch.stack(ptxt, 1).cpu().numpy().astype(np.float16)  # (P, 48, T)
            b["head_text_share"] = np.array(tshare, np.float32)
            b["head_dist"] = np.array(mdist, np.float32)
            b["head_entropy"] = np.array(ent, np.float32)
            return {}

        def capture_weights(dm):
            wd = {}
            err = []
            for i, blk in enumerate(dm.blocks):
                a, mlp = blk.attn, blk.mlp
                for nm, lin in (("wq", a.wq), ("wk", a.wk), ("wv", a.wv), ("gate", a.gate), ("wo", a.wo),
                                ("mlp_gate", mlp.gate), ("mlp_up", mlp.up), ("mlp_down", mlp.down)):
                    wt = lin.weight
                    try:
                        wt = wt.to("cuda")
                    except Exception:
                        pass
                    w = _deq(wt).to("cuda")
                    if i == 0:                                       # check dequantized W reproduces the layer
                        x = torch.randn(4, w.shape[1], device="cuda", dtype=torch.bfloat16)
                        try:
                            ref = lin(x).float()
                            err.append([nm, float((ref - x.float() @ w.T).norm() / ref.norm())])
                        except Exception as e:
                            err.append([nm, str(e)[:80]])
                    wd[f"{i}_{nm}"] = _thumb(w)
                    if nm in ("wq", "wo"):
                        r = w.reshape(48, 128, -1) if nm == "wq" else w.T.reshape(48, 128, -1)
                        wd[f"{i}_{nm}_headnorm"] = r.norm(dim=(1, 2)).cpu().numpy()
                    if nm in ("wk", "wv"):
                        wd[f"{i}_{nm}_headnorm"] = w.reshape(12, 128, -1).norm(dim=(1, 2)).cpu().numpy()
                    if nm == "mlp_down":
                        wd[f"{i}_neuron_out_norm"] = w.norm(dim=0).cpu().numpy().astype(np.float16)
                    del w
            pj = _deq(dm.txtfusion.projector.weight).flatten().cpu().numpy()
            np.savez_compressed(f"{out_dir}/weights.npz", **wd)
            meta["txtfusion_mix"] = pj.tolist()
            meta["weight_check"] = err

        def wrapper(apply_model, args):
            dm = m.model.diffusion_model
            st["step"] += 1
            st["cur"] = {"blocks": {}, "n_txt": 0}
            st["pca_prev"] = None
            if not st["weights_done"]:
                with torch.no_grad():
                    capture_weights(dm)
                st["weights_done"] = True
            ctx = args["c"].get("c_crossattn")
            if ctx is not None and st["step"] == 0:
                c = ctx[0].float().reshape(ctx.shape[1], 12, -1)
                np.save(f"{out_dir}/textenc_layer_norm.npy", c.norm(dim=-1).cpu().numpy())
            x = args["input"]
            hs = hooks_for(dm)
            try:
                out = apply_model(x, args["timestep"], **args["c"])
            finally:
                for h in hs:
                    h.remove()
            s = st["step"]
            np.save(f"{out_dir}/xt_{s}.npy", m.model.process_latent_out(x.detach().float()).cpu().numpy())
            np.save(f"{out_dir}/x0_{s}.npy", m.model.process_latent_out(out.detach().float()).cpu().numpy())
            cur = st["cur"]
            pack = {"embed_norm": cur["embed_norm"], "txt_norm": cur["txt_norm"]}
            for i, b in cur["blocks"].items():
                for k2, v2 in b.items():
                    pack[f"b{i}_{k2}"] = v2
            np.savez_compressed(f"{out_dir}/step_{s}.npz", **pack)
            meta["steps"].append({"sigma": float(args["timestep"].flatten()[0]), "n_txt": cur["n_txt"]})
            json.dump(meta, open(f"{out_dir}/meta.json", "w"))
            st["cur"] = None
            return out

        m.set_model_unet_function_wrapper(wrapper)
        m.set_model_attn1_patch(attn_patch)
        return (m,)
