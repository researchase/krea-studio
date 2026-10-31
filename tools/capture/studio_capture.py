"""Capture one real generation end to end for Krea Studio (Godot): book above the head, seed 23."""
import json, os, numpy as np, kvtest, leadlag_gen as LG
OUT = "/mnt/nvme1tb/ComfyUI/custom_nodes/rs_kvshare/experiments/studio_capture"
NAME, SEED = "head_s23", 23
action = f"{LG.PRE} {LG.HEIGHTS['head']}"
kp = json.load(open(f"leadlag/{NAME}_kp.json"))[-1]["people"]
b = np.array(max(kp, key=lambda p: sum(p["pose_keypoints_2d"][2::3]))["pose_keypoints_2d"]).reshape(-1, 3)
row = [r for r in np.load("leadlag/analysis.npy", allow_pickle=True) if r["name"] == NAME][0]
P = lambda j: [int(b[j, 1] // 16), int(b[j, 0] // 16)]
probes = [["book", int(round(row["by"])), int(round(row["bx"]))], ["right hand", *P(4)], ["left hand", *P(7)],
          ["right elbow", *P(3)], ["left elbow", *P(6)], ["right shoulder", *P(2)], ["face", *P(0)],
          ["chest", int((b[1, 1] + b[8, 1]) / 32), int((b[1, 0] + b[8, 0]) / 32)], ["wall", 8, 10], ["floor", 44, 20]]
print(probes)
g = kvtest.graph(action, "off", "studio_" + NAME, seed=SEED)
g["kv"] = {"class_type": "RSCapture", "inputs": {"model": ["l2", 0], "out_dir": OUT, "probes": json.dumps(probes), "weights": True}}
print(kvtest.run(g))
for kind in ("x0", "xt"):
    np.save(f"{OUT}/{kind}_all.npy", np.concatenate([np.load(f"{OUT}/{kind}_{s}.npy") for s in range(8)]))
    print(kind, kvtest.run({
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "lat": {"class_type": "RSLoadLatentNpy", "inputs": {"path": f"{OUT}/{kind}_all.npy"}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["lat", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": f"studio/{kind}"}}}))
