# Krea Studio

One **real** Krea 2 generation, prompt to picture, replayed in 3D. Nothing is simulated: the data was recorded from
the real 28-block model (real weights) by the `RSCapture` node in `/mnt/nvme1tb/ComfyUI/custom_nodes/rs_kvshare/`
(`experiments/studio_capture.py`), then packed by `tools/export_data.py`. The picture: "...holding a red book with both
hands high above his head", seed 23 (the same picture as `leadlag/head_s23`, so its SDPose tracking lines up exactly).

Run: `~/apps/Godot_v4.7.2-stable_linux.x86_64 --path ~/godot/KreaStudio`
Screenshot tour (no person needed): add `-- --shot=/some/dir`.

## The hall (left to right = the computation)
1. **Prompt** → Qwen3-VL-4B (12 layers tapped; rows) → text fusion (its real learned 12-layer mix) → 115 text tokens.
   Tokens glow by how much the selected probe reads them in the selected block/head.
2. **Noisy input x_t** of the current step → 2×2 patches → 48 × 84 = 4032 image tokens.
3. **28 blocks.** Each: the big panel (choose the view), a **rack of all 48 heads** (12 columns = the 12 shared
   key/value heads, 4 query heads each; each tile = that head's attention heat map for the probe; gold frame = reads the
   prompt, blue = reads the picture), bars = what the probe reads in the prompt, the 16384-neuron **MLP wall**, the
   **real weights** (|W| thumbnails of the dequantized base weights), and the attention/MLP gate bars (adaLN).
4. **Preview**: velocity → x0 guess (what the model thinks the final picture is at this step), then the Euler step.
5. **The picture** after 8 steps.
6. **Tracking tower**: the 8 previews in depth with shoulders, elbows, wrists, book and hands threaded through time.

Panel views: residual stream (PCA colours), token norm, **keys & locks** (blue = attention write, orange = MLP write,
white = both: Ashwin's lock-and-key hypothesis), attention write, MLP write, probe attention (all heads), selected head,
selected neuron, the picture. Overlays on every picture: word maps (character/holding/book/hands/where), body regions,
background, clicked-patch attention. Tracking toggles: skeleton, hands (21 points each), wrists, elbows, shoulders,
face, book, hands-word centroid, trajectories, probe ring.

**3D**: "3D columns" turns every patch into a column (height = the value of the chosen view), each head's heat map
into a little 3D terrain of columns, and the MLP wall into a surface spiking where neurons fire.
**Flow: layer to layer** stacks the 28 blocks in depth, so the residual stream runs through them like a corridor.

**Influence lines (token to token)**:
- *Token → token*: from the probe's patch to the 8 patches and 3 words it reads most (in the flow view, the patches
  sit on the previous block, because block b's attention reads what block b-1 wrote).
- *Body chain*: book, hands, elbows, shoulder, face, chest; the 12 strongest "who reads whom" links per block
  (x its share if attention were uniform); in the flow view from block b to block b-1.
- *Joints*: SDPose fingertips (thumb, index, middle, little), wrists, elbows, shoulders, face and the book at the
  current step, linked by how strongly one joint's patch attends to the other's (mean over heads, blocks 4/10/16/22).
Thickness and brightness = strength. In the hall the lines arc out of the block.

Controls: drag = look, WASD = fly, Alt+drag = orbit, click = pick (head tile, neuron, weight, any patch),
Space = play the generation, ←/→ = step.

## Honest limits
- Head attention is stored for 10 probe points (book, both hands, both elbows, right shoulder, face, chest, wall,
  floor); "clicked patch" attention for any patch is the mean over heads of blocks 4/10/16/22 (top 32 cells).
- Head maps are pooled to 4×4-patch cells; neuron maps exist only for each block's 16 most place-selective neurons.
- Weight thumbnails: dequantized int8 base weights (LoRAs not merged); a check reproduced the layers to 8-18 %.
- SDPose finds no confident joints in the blurriest first steps, so tracks start at step 2-3.
