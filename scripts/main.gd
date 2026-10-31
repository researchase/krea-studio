extends Node3D

## Krea Studio: one REAL Krea 2 generation, prompt to picture, replayed from recorded activations.
##
## Nothing here is simulated. tools/export_data.py packs what the rs_kvshare RSCapture node recorded while the real
## 28-block model (real weights) drew one picture: every block's residual stream, all 48 heads' attention for ten
## probe points, the 16384-neuron MLPs, the gates, and thumbnails of the dequantized weights. Godot only replays it.
##
## The hall runs left to right, the way the computation does:
##   prompt -> Qwen3-VL (12 layers tapped) -> text tokens | noise x_t -> 4032 patch tokens
##   -> 28 single-stream blocks (attention: 48 query heads on 12 key/value heads; SwiGLU MLP)
##   -> velocity -> x0 guess (the preview) -> Euler step -> next x_t ... x 8 steps -> picture
## At the end stands the tracking tower: the eight previews in depth, with the joints threaded through them.

const D := "res://data/"
const ROOM_BG := Color(0.078, 0.047, 0.024)
const PW := 8.4
const PH := 4.8
const HALL_X0 := 30.0
const BLOCK_DX := 12.0
const IMG_W := 1344.0
const IMG_H := 768.0

const VIEWS := ["Residual stream (PCA colours)", "Token size (norm)", "Keys & locks (attention vs MLP write)",
	"Attention write", "MLP write", "Probe attention (all heads)", "Selected head", "Selected neuron",
	"Picture (this step's guess)"]
const LINE_MODES := ["None", "Token → token (what the probe reads)", "Body chain: probe ↔ probe (per block)",
	"Joints: fingers, wrists, elbows, shoulders, book"]
const BODY_PROBES := 8          ## book, both hands, both elbows, shoulder, face, chest (wall and floor left out)
const OVERLAYS := ["None", "Word: character", "Word: holding", "Word: book", "Word: hands", "Word: where",
	"Body regions", "Background", "Clicked patch attention"]
const REGION_COLS := [Color(0.25, 0.3, 0.4), Color(0.85, 0.75, 0.6), Color(1.0, 0.55, 0.7), Color(0.3, 0.85, 0.35),
	Color(0.3, 0.6, 1.0), Color(0.4, 0.95, 1.0), Color(1.0, 0.8, 0.2), Color(1.0, 0.2, 0.55)]
const GROUP_COLS := {"char": Color(1.0, 0.62, 0.3), "holding": Color(1.0, 0.8, 0.2), "book": Color(1.0, 0.3, 0.45),
	"hands": Color(0.4, 0.95, 1.0), "where": Color(0.4, 1.0, 0.5)}
const JOINTS := {"shoulders": [[2, 5], Color(0.3, 0.9, 0.35)], "elbows": [[3, 6], Color(0.3, 0.6, 1.0)],
	"wrists": [[4, 7], Color(1.0, 0.6, 0.15)], "face": [[0, 14, 15, 16, 17], Color(1.0, 0.55, 0.75)]}

var meta: Dictionary
var S := 8
var B := 28
var P := 10
var NH := 48
var T := 115
var bin := {}

var step := 0
var probe := 0
var view := 0
var overlay := 0
var sel_block := 14
var sel_head := 0
var sel_neuron := -1          ## slot 0..15 in the selected block's top-16 list
var picked_patch := -1
var show := {"skeleton": true, "hands": true, "wrists": true, "elbows": true, "shoulders": true, "face": false,
	"book": true, "hands word": false, "trajectories": true, "probe": true}
var playing := false
var play_x := 0.0
var speed := 1.0

var cam: OrbitCamera
var shader: Shader
var quad: QuadMesh
var blocks := []              ## per block: Dictionary of materials and nodes
var overlay_mat: ShaderMaterial
var noise_mat: ShaderMaterial
var preview_mat: ShaderMaterial
var tower_mats := []
var tower_over := []
var token_labels := []
var track_mesh: ImmediateMesh
var track_points: MultiMesh
var probe_mesh: ImmediateMesh
var tower_root: Node3D
var pulse: MeshInstance3D
var out_x := 0.0
var tower_x := 0.0
var pickables := []           ## {kind, rect: Rect2 in world XY, z, data}
var ui: Control
var info: RichTextLabel
var step_label: Label
var step_slider: HSlider
var block_slider: HSlider
var press_pos := Vector2.ZERO
var lines_mode := 2          ## 0 none, 1 token->token, 2 body chain (probes), 3 joints (SDPose + patch attention)
var show3d := true            ## data as columns / terrains instead of flat textures
var flow := false             ## blocks stacked in depth: the residual stream flows layer to layer
var cells_shader: Shader
var terrain_shader: Shader
var ident_252 := PackedFloat32Array()
var ident_4032 := PackedFloat32Array()
var lines_root: Node3D
var hall_rail: MeshInstance3D
var flow_rail: MeshInstance3D
var noise_root: Node3D
var out_root: Node3D
const FLOW_DZ := 4.0
const COL_H := 1.2
var ui_layer: CanvasLayer
var flow_box: CheckBox
var lines_menu: OptionButton
var view_menu: OptionButton
var overlay_menu: OptionButton
var probe_menu: OptionButton
var pic_tex := []             ## x0 ImageTextures
var xt_tex := []


func _ready() -> void:
	RenderingServer.set_default_clear_color(ROOM_BG)
	meta = JSON.parse_string(FileAccess.get_file_as_string(D + "meta.json"))
	var sh: Dictionary = meta["shape"]
	S = int(sh["S"]); B = int(sh["B"]); P = int(sh["P"]); NH = int(sh["heads"]); T = int(sh["T"])
	for f in meta["files"]:
		bin[f] = FileAccess.get_file_as_bytes(D + f)
	for s in S:
		pic_tex.append(_load_pic("img/x0_%d.jpg" % s))
		xt_tex.append(_load_pic("img/xt_%d.jpg" % s))
	shader = load("res://shaders/view.gdshader")
	cells_shader = load("res://shaders/cells.gdshader")
	terrain_shader = load("res://shaders/terrain.gdshader")
	quad = QuadMesh.new()
	quad.size = Vector2(1, 1)
	cam = $Camera
	_build_world()
	_build_ui()
	_refresh_all()
	cam.fly_to(Vector3(HALL_X0 + 6, 2.0, 30.0), Vector3(HALL_X0 + 6, 0.5, 0.0))
	var args := OS.get_cmdline_user_args()
	for a in args:
		if a.begins_with("--shot="):
			_auto_shot(a.substr(7))
		if a.begins_with("--anim="):
			_auto_anim(a.substr(7))


# ------------------------------------------------------------------------------------------------ data helpers

func _load_pic(rel: String) -> ImageTexture:
	var img := Image.load_from_file(ProjectSettings.globalize_path(D + rel))
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


func _tex(name: String, off: int, w: int, h: int, fmt := Image.FORMAT_L8) -> ImageTexture:
	var ch := 3 if fmt == Image.FORMAT_RGB8 else 1
	var bytes: PackedByteArray = bin[name].slice(off, off + w * h * ch)
	return ImageTexture.create_from_image(Image.create_from_data(w, h, false, fmt, bytes))


func _f32(name: String, idx: int) -> float:
	return bin[name].decode_float(idx * 4)


func _mat(mode: int, opacity := 1.0) -> ShaderMaterial:
	var m := ShaderMaterial.new()
	m.shader = shader
	m.set_shader_parameter("mode", mode)
	m.set_shader_parameter("opacity", opacity)
	var pal := PackedVector3Array()
	for c in REGION_COLS:
		pal.append(Vector3(c.r, c.g, c.b))
	m.set_shader_parameter("pal", pal)
	if opacity < 1.0:
		m.render_priority = 2
	return m


func _plane(parent: Node3D, pos: Vector3, size: Vector2, mat: Material) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = quad
	mi.scale = Vector3(size.x, size.y, 1)
	mi.position = pos
	mi.material_override = mat
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	parent.add_child(mi)
	return mi


func _label(parent: Node3D, text: String, pos: Vector3, size := 48, col := Color(1, 0.9, 0.75), left := false) -> Label3D:
	var l := Label3D.new()
	l.text = text
	l.font_size = size
	l.pixel_size = 0.005
	l.modulate = col
	l.outline_size = 8
	l.outline_modulate = Color(0, 0, 0, 0.8)
	l.position = pos
	l.shaded = false
	l.double_sided = true
	if left:
		l.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	parent.add_child(l)
	return l


func _box(parent: Node3D, pos: Vector3, size: Vector3, col: Color, emit := 0.0) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	mi.mesh = bm
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	if emit > 0.0:
		m.emission_enabled = true
		m.emission = col
		m.emission_energy_multiplier = emit
	mi.material_override = m
	mi.position = pos
	parent.add_child(mi)
	return mi


func _pick(kind: String, root: Node3D, local: Vector3, size: Vector2, data := {}) -> void:
	pickables.append({"kind": kind, "root": root, "local": local, "size": size, "data": data})


func _ident(n: int) -> PackedFloat32Array:
	var a := PackedFloat32Array()
	a.resize(n * 12)
	for i in n:
		a[i * 12] = 1.0
		a[i * 12 + 5] = 1.0
		a[i * 12 + 10] = 1.0
	return a


## a field of columns: one MultiMesh of unit boxes, placed and coloured on the GPU by cells.gdshader
func _cells(parent: Node3D, pos: Vector3, gw: int, gh: int, size: Vector2, h: float) -> Dictionary:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	var bm := BoxMesh.new()
	bm.size = Vector3.ONE
	mm.mesh = bm
	mm.instance_count = gw * gh
	if gw * gh == 252:
		if ident_252.is_empty():
			ident_252 = _ident(252)
		mm.buffer = ident_252
	else:
		if ident_4032.is_empty():
			ident_4032 = _ident(4032)
		mm.buffer = ident_4032
	var mi := MultiMeshInstance3D.new()
	mi.multimesh = mm
	var m := ShaderMaterial.new()
	m.shader = cells_shader
	m.set_shader_parameter("grid_w", gw)
	m.set_shader_parameter("grid_h", gh)
	m.set_shader_parameter("size", size)
	m.set_shader_parameter("height", h)
	mi.material_override = m
	mi.position = pos
	mi.custom_aabb = AABB(Vector3(-size.x / 2, -size.y / 2, -0.1), Vector3(size.x, size.y, h * 1.3 + 0.2))
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	parent.add_child(mi)
	return {"mi": mi, "mat": m}


func _terrain(parent: Node3D, pos: Vector3, w: float) -> Dictionary:
	var pm := PlaneMesh.new()
	pm.size = Vector2(w, w)
	pm.subdivide_width = 127
	pm.subdivide_depth = 127
	pm.orientation = PlaneMesh.FACE_Z
	var mi := MeshInstance3D.new()
	mi.mesh = pm
	var m := ShaderMaterial.new()
	m.shader = terrain_shader
	mi.material_override = m
	mi.position = pos
	mi.custom_aabb = AABB(Vector3(-w / 2, -w / 2, -0.1), Vector3(w, w, 1.2))
	parent.add_child(mi)
	return {"mi": mi, "mat": m}


func _pick_rect(pk: Dictionary) -> Rect2:
	var c: Vector3 = (pk["root"] as Node3D).position + pk["local"]
	var sz: Vector2 = pk["size"]
	return Rect2(c.x - sz.x / 2, c.y - sz.y / 2, sz.x, sz.y)


func _pick_z(pk: Dictionary) -> float:
	return (pk["root"] as Node3D).position.z + (pk["local"] as Vector3).z


# ------------------------------------------------------------------------------------------------ the hall

func _build_world() -> void:
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, 30, 0)
	sun.light_energy = 0.8
	add_child(sun)
	# the residual-stream rail: every station hangs off it, the pulse rides it
	out_x = HALL_X0 + B * BLOCK_DX + 6.0
	tower_x = out_x + 34.0
	hall_rail = _box(self, Vector3((out_x - 14.0) / 2.0, 0.0, -1.5), Vector3(out_x + 18.0, 0.08, 0.08),
		Color(1.0, 0.55, 0.2), 2.5)
	flow_rail = _box(self, Vector3(HALL_X0, -PH / 2 - 0.3, -(B + 1) * FLOW_DZ / 2.0), Vector3(0.08, 0.08, (B + 3) * FLOW_DZ),
		Color(1.0, 0.55, 0.2), 2.5)
	flow_rail.visible = false
	lines_root = Node3D.new()
	add_child(lines_root)
	pulse = _box(self, Vector3(0, 0.0, -1.5), Vector3(0.6, 0.35, 0.35), Color(1.0, 0.85, 0.5), 6.0)
	track_mesh = ImmediateMesh.new()
	probe_mesh = ImmediateMesh.new()
	track_points = MultiMesh.new()
	track_points.transform_format = MultiMesh.TRANSFORM_3D
	track_points.use_colors = true
	var sp := SphereMesh.new()
	sp.radius = 0.06
	sp.height = 0.12
	sp.radial_segments = 8
	sp.rings = 4
	track_points.mesh = sp
	overlay_mat = _mat(4, 0.8)
	_build_prompt()
	_build_noise()
	for b in B:
		_build_block(b)
	_build_output()
	_build_tower()


func _track_layer(parent: Node3D, z: float) -> Array:
	## the same marker resources are shared by every big plane, so one rebuild moves them all
	var lm := StandardMaterial3D.new()
	lm.vertex_color_use_as_albedo = true
	lm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	lm.no_depth_test = false
	var li := MeshInstance3D.new()
	li.mesh = track_mesh
	li.material_override = lm
	li.position = Vector3(0, 0, z)
	parent.add_child(li)
	var pi := MeshInstance3D.new()
	pi.mesh = probe_mesh
	pi.material_override = lm
	pi.position = Vector3(0, 0, z)
	parent.add_child(pi)
	var mm := MultiMeshInstance3D.new()
	mm.multimesh = track_points
	var pm := StandardMaterial3D.new()
	pm.vertex_color_use_as_albedo = true
	pm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mm.material_override = pm
	mm.position = Vector3(0, 0, z)
	parent.add_child(mm)
	return [li, pi, mm]


func _build_prompt() -> void:
	var root := Node3D.new()
	root.position = Vector3(0, 0, 0)
	add_child(root)
	_label(root, "1 · PROMPT → Qwen3-VL-4B → TEXT TOKENS", Vector3(0, PH / 2 + 4.3, 0), 80)
	# the tokens, wrapped, coloured by the word groups the experiments care about
	var x := -7.5
	var y := PH / 2 + 3.4
	var gspan: Dictionary = meta["groups"]
	for i in T:
		var t: String = meta["tokens"][i]
		var col := Color(0.7, 0.62, 0.55)
		for g in gspan:
			if i >= int(gspan[g][0]) and i <= int(gspan[g][1]):
				col = GROUP_COLS.get(g, col)
		var w := 0.15 * maxf(t.length(), 2)
		if x + w > 7.5:
			x = -7.5
			y -= 0.5
		var l := _label(root, t.strip_edges() if t != "<end>" else "·", Vector3(x + w / 2, y, 0), 64, col)
		token_labels.append(l)
		x += w + 0.05
	# 12 Qwen3-VL layers tapped: per-token norm per layer (rows = layers)
	var tl: Array = meta["textenc_layer_norm"]
	var img := Image.create_empty(T, 12, false, Image.FORMAT_L8)
	for layer in 12:
		var mx := 0.0
		var mn := INF
		for i in T:
			mx = maxf(mx, float(tl[i][layer]))
			mn = minf(mn, float(tl[i][layer]))
		for i in T:
			img.set_pixel(i, layer, Color((float(tl[i][layer]) - mn) / maxf(mx - mn, 1e-6), 0, 0))
	var m := _mat(1)
	m.set_shader_parameter("data", ImageTexture.create_from_image(img))
	_plane(root, Vector3(0, -0.6, 0), Vector2(13.0, 2.6), m)
	_label(root, "Qwen3-VL text encoder: 12 layers tapped (rows) × 115 tokens (columns), brightness = activation size",
		Vector3(0, 1.0, 0), 30)
	# txtfusion's learned mix of the 12 layers (a real weight)
	var mix: Array = meta["txtfusion_mix"]
	for layer in 12:
		var v := float(mix[layer])
		var h := absf(v) * 1.2 + 0.05
		_box(root, Vector3(-3.3 + layer * 0.6, -2.6 + (h / 2 if v > 0 else -h / 2), 0),
			Vector3(0.4, h, 0.2), Color(1.0, 0.6, 0.2) if v > 0 else Color(0.3, 0.55, 1.0), 1.0)
	_label(root, "text fusion: learned mix of the 12 layers (orange +, blue −) → 2 refiner blocks → 115 tokens × 6144",
		Vector3(0, -4.0, 0), 30)
	_pick("station", root, Vector3.ZERO, Vector2(14, 10), {"name": "prompt"})


func _build_noise() -> void:
	var root := Node3D.new()
	root.position = Vector3(15.0, 0, 0)
	add_child(root)
	noise_mat = _mat(0)
	_plane(root, Vector3.ZERO, Vector2(PW, PH), noise_mat)
	var ov := _plane(root, Vector3(0, 0, 0.02), Vector2(PW, PH), overlay_mat)
	ov.name = "overlay"
	_track_layer(root, 0.05)
	_label(root, "2 · NOISY INPUT x_t → patchify 2×2 → 48 × 84 = 4032 image tokens", Vector3(0, PH / 2 + 0.5, 0), 56)
	_pick("plane", root, Vector3.ZERO, Vector2(PW, PH), {"name": "noise"})
	noise_root = root


func _build_block(b: int) -> void:
	var bx := HALL_X0 + b * BLOCK_DX
	var root := Node3D.new()
	root.position = Vector3(bx, 0, 0)
	add_child(root)
	var d := {"root": root}
	d["panel"] = _mat(3)
	d["panel_mi"] = _plane(root, Vector3.ZERO, Vector2(PW, PH), d["panel"])
	d["overlay_mi"] = _plane(root, Vector3(0, 0, 0.02), Vector2(PW, PH), overlay_mat)
	d["track"] = _track_layer(root, 0.05)
	d["cells"] = _cells(root, Vector3.ZERO, 84, 48, Vector2(PW, PH), COL_H)
	_pick("panel", root, Vector3.ZERO, Vector2(PW, PH), {"b": b})
	d["title"] = _label(root, "BLOCK %d" % b, Vector3(0, -PH / 2 - 0.35, 0.05), 64)
	# head rack: 12 columns = the 12 key/value heads, 4 rows = the query heads that share each one
	d["tiles"] = []
	d["frames"] = []
	for h in NH:
		var g := h / 4
		var r := h % 4
		var c := Vector3(-PW / 2 + (g + 0.5) * 0.7, PH / 2 + 0.45 + (3 - r + 0.5) * 0.42, 0)
		var fm := StandardMaterial3D.new()
		fm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		_plane(root, c + Vector3(0, 0, -0.01), Vector2(0.68, 0.40), fm)
		d["frames"].append(fm)
		var tm := _mat(1)
		_plane(root, c, Vector2(0.62, 0.35), tm)
		d["tiles"].append(tm)
		d.get_or_add("tile_cells", []).append(_cells(root, c + Vector3(0, 0, 0.01), 21, 12, Vector2(0.62, 0.35), 0.28))
		_pick("head", root, c, Vector2(0.68, 0.40), {"b": b, "h": h})
	_label(root, "48 heads · columns = 12 shared key/value heads · gold frame = reads the prompt, blue = reads the picture",
		Vector3(0, PH / 2 + 2.35, 0), 26)
	# what the probe reads in the prompt: one bar per text token
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	var bm := BoxMesh.new()
	bm.size = Vector3(1, 1, 1)
	mm.mesh = bm
	mm.instance_count = T
	var mmi := MultiMeshInstance3D.new()
	mmi.multimesh = mm
	var um := StandardMaterial3D.new()
	um.vertex_color_use_as_albedo = true
	um.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mmi.material_override = um
	root.add_child(mmi)
	d["txtbars"] = mm
	# MLP neuron wall (16384 = 128 x 128) and the real weights
	d["wall"] = _mat(2)
	d["wall_flat"] = _plane(root, Vector3(-PW / 2 + 1.6, -PH / 2 - 2.35, 0), Vector2(3.2, 3.2), d["wall"])
	d["wall3d"] = _terrain(root, Vector3(-PW / 2 + 1.6, -PH / 2 - 2.35, 0), 3.2)
	_pick("wall", root, Vector3(-PW / 2 + 1.6, -PH / 2 - 2.35, 0), Vector2(3.2, 3.2), {"b": b})
	_label(root, "MLP: 16384 neurons", Vector3(-PW / 2 + 1.6, -PH / 2 - 4.15, 0), 28)
	var mats: Array = meta["weight_mats"]
	for k in 8:
		var wc := Vector3(-PW / 2 + 3.9 + (k % 4) * 1.05, -PH / 2 - 1.55 - (k / 4) * 1.35, 0)
		var wm := _mat(1)
		wm.set_shader_parameter("data", _tex("weights.bin", (b * 8 + k) * 4096, 64, 64))
		_plane(root, wc, Vector2(0.95, 0.95), wm)
		_label(root, mats[k], wc + Vector3(0, -0.6, 0), 24)
		_pick("weight", root, wc, Vector2(0.95, 0.95), {"b": b, "k": k})
	_label(root, "real weights |W| (dequantized)", Vector3(-PW / 2 + 5.5, -PH / 2 - 4.15, 0), 26)
	# adaLN gates: how loudly attention and the MLP are allowed to write at this noise level
	d["gate_a"] = _box(root, Vector3(PW / 2 - 0.35, -PH / 2 - 2.0, 0), Vector3(0.25, 1, 0.25), Color(0.35, 0.6, 1.0), 1.5)
	d["gate_m"] = _box(root, Vector3(PW / 2 - 0.05, -PH / 2 - 2.0, 0), Vector3(0.25, 1, 0.25), Color(1.0, 0.6, 0.2), 1.5)
	blocks.append(d)


func _build_output() -> void:
	var root := Node3D.new()
	root.position = Vector3(out_x, 0, 0)
	add_child(root)
	preview_mat = _mat(0)
	_plane(root, Vector3.ZERO, Vector2(PW, PH), preview_mat)
	_plane(root, Vector3(0, 0, 0.02), Vector2(PW, PH), overlay_mat)
	_track_layer(root, 0.05)
	_label(root, "3 · LAST LAYER → velocity → x0 GUESS (the preview)", Vector3(0, PH / 2 + 0.5, 0), 56)
	_label(root, "Euler step: x_next = x_t + (σ_next − σ) · velocity  → back to block 0 for the next step",
		Vector3(0, -PH / 2 - 0.5, 0), 34)
	_pick("plane", root, Vector3.ZERO, Vector2(PW, PH), {"name": "preview"})
	out_root = root
	var fm := _mat(0)
	fm.set_shader_parameter("pic", _load_pic("img/final.jpg"))
	_plane(root, Vector3(16.0, 0, 0), Vector2(PW * 1.6, PH * 1.6), fm)
	_label(root, "4 · THE PICTURE (after 8 steps, VAE decode)", Vector3(16.0, PH * 0.8 + 0.5, 0), 56)


func _build_tower() -> void:
	tower_root = Node3D.new()
	tower_root.position = Vector3(tower_x, 0, 0)
	add_child(tower_root)
	_label(tower_root, "5 · TRACKING TOWER: the 8 previews in depth (step 1 in front); joints threaded through time",
		Vector3(0, PH / 2 + 0.8, 0), 52)
	for s in S:
		var m := _mat(0, 0.55)                      # see-through, so the joint paths show through the stack
		m.set_shader_parameter("pic", pic_tex[s])
		var z := -s * 2.6
		_plane(tower_root, Vector3(0, 0, z), Vector2(PW, PH), m)
		var o := _mat(4, 0.75)
		_plane(tower_root, Vector3(0, 0, z + 0.02), Vector2(PW, PH), o)
		tower_mats.append(m)
		tower_over.append(o)
		_label(tower_root, "step %d" % (s + 1), Vector3(-PW / 2 - 0.6, PH / 2 - 0.3, z), 40)


# ------------------------------------------------------------------------------------------------ refresh

func _sync_menus() -> void:
	if view_menu:
		view_menu.select(view)
		overlay_menu.select(overlay)
		probe_menu.select(probe)
		lines_menu.select(lines_mode)


func _refresh_all() -> void:
	_sync_menus()
	_apply_3d()
	noise_mat.set_shader_parameter("pic", xt_tex[step])
	preview_mat.set_shader_parameter("pic", pic_tex[step])
	_refresh_blocks()
	_refresh_heads()
	_refresh_overlay()
	_refresh_tracking()
	_refresh_tokens()
	_refresh_lines()
	_refresh_info()
	if step_label:
		var sig: float = meta["steps"][step]["sigma"]
		step_label.text = "Step %d of 8   (noise level σ = %.2f)" % [step + 1, sig]


func _off(name: String, per: int, idx: int) -> int:
	return idx * per


func _refresh_blocks() -> void:
	for b in B:
		var d: Dictionary = blocks[b]
		var m: ShaderMaterial = d["panel"]
		var sb := step * B + b
		match view:
			0:
				m.set_shader_parameter("mode", 3)
				m.set_shader_parameter("gain", 1.6)
				m.set_shader_parameter("data", _tex("pca.bin", sb * 4032 * 3, 84, 48, Image.FORMAT_RGB8))
			1, 3, 4:
				var f: String = ["", "norm.bin", "", "write_attn.bin", "write_mlp.bin"][view]
				m.set_shader_parameter("mode", 1)
				m.set_shader_parameter("gain", 1.0)
				m.set_shader_parameter("data", _tex(f, sb * 4032, 84, 48))
			2:
				var a: PackedByteArray = bin["write_attn.bin"].slice(sb * 4032, sb * 4032 + 4032)
				var l: PackedByteArray = bin["write_mlp.bin"].slice(sb * 4032, sb * 4032 + 4032)
				var rgb := PackedByteArray()
				rgb.resize(4032 * 3)
				for i in 4032:
					rgb[i * 3] = l[i]                       # orange-red = MLP (the lock opens)
					rgb[i * 3 + 1] = mini(a[i], l[i])       # both = white-ish
					rgb[i * 3 + 2] = a[i]                   # blue = attention (the key)
				m.set_shader_parameter("mode", 3)
				m.set_shader_parameter("gain", 1.0)
				m.set_shader_parameter("data", ImageTexture.create_from_image(
					Image.create_from_data(84, 48, false, Image.FORMAT_RGB8, rgb)))
			5, 6:
				m.set_shader_parameter("mode", 1)
				m.set_shader_parameter("gain", 1.0)
				m.set_shader_parameter("data", _probe_map(b, -1 if view == 5 else sel_head))
			7:
				var slot := sel_neuron if (b == sel_block and sel_neuron >= 0) else 0
				m.set_shader_parameter("mode", 2)
				m.set_shader_parameter("gain", 1.0)
				m.set_shader_parameter("data", _tex("neuron_maps.bin", (sb * 16 + slot) * 4032, 84, 48))
			8:
				m.set_shader_parameter("mode", 0)
				m.set_shader_parameter("pic", pic_tex[step])
		var cm: ShaderMaterial = d["cells"]["mat"]
		cm.set_shader_parameter("data", m.get_shader_parameter("data"))
		cm.set_shader_parameter("pic", pic_tex[step])
		cm.set_shader_parameter("mode", [3, 1, 4, 1, 1, 1, 1, 2, 0][view])
		cm.set_shader_parameter("gain", 1.6 if view == 0 else 1.0)
		var wt := _tex("neuron_mean.bin", sb * 16384, 128, 128)
		(d["wall"] as ShaderMaterial).set_shader_parameter("data", wt)
		(d["wall3d"]["mat"] as ShaderMaterial).set_shader_parameter("data", wt)
		var mod: Array = meta["mod"][step][b]
		var ga: MeshInstance3D = d["gate_a"]
		var gm: MeshInstance3D = d["gate_m"]
		ga.scale.y = clampf(float(mod[2]) * 3.0, 0.05, 2.5)
		gm.scale.y = clampf(float(mod[5]) * 3.0, 0.05, 2.5)
		var lbl: Label3D = d["title"]
		lbl.text = "BLOCK %d   ·   keys↔locks r = %+.2f" % [b, float(meta["keylock_r"][step][b])]
		lbl.modulate = Color(1, 0.95, 0.6) if b == sel_block else Color(1, 0.9, 0.75)


## a probe's attention mass over the picture's 12 x 21 cells (4x4 patches), one head or the sum of all 48
func _probe_cells(b: int, h: int, pr := -1) -> PackedFloat32Array:
	var base := ((step * B + b) * P + (probe if pr < 0 else pr)) * NH
	var acc := PackedFloat32Array()
	acc.resize(252)
	var heads: Array = range(NH) if h < 0 else [h]
	for hh in heads:
		var o: int = (base + hh) * 252
		var scale := _f32("att_img_max.bin", base + hh)
		for i in 252:
			acc[i] += bin["att_img.bin"][o + i] / 255.0 * scale
	return acc


## the same, as a texture for the panel
func _probe_map(b: int, h: int) -> ImageTexture:
	var acc := _probe_cells(b, h)
	var mx := 0.0
	for i in 252:
		mx = maxf(mx, acc[i])
	var img := Image.create_empty(21, 12, false, Image.FORMAT_L8)
	for i in 252:
		img.set_pixel(i % 21, i / 21, Color(acc[i] / maxf(mx, 1e-9), 0, 0))
	return ImageTexture.create_from_image(img)


func _refresh_heads() -> void:
	for b in B:
		var d: Dictionary = blocks[b]
		var base := ((step * B + b) * P + probe) * NH
		for h in NH:
			var tm: ShaderMaterial = d["tiles"][h]
			var tt := _tex("att_img.bin", (base + h) * 252, 21, 12)
			tm.set_shader_parameter("data", tt)
			(d["tile_cells"][h]["mat"] as ShaderMaterial).set_shader_parameter("data", tt)
			var ts := _f32("att_txt_sum.bin", base + h)          # share of this probe's attention on the prompt
			var fm: StandardMaterial3D = d["frames"][h]
			var c := Color(0.25, 0.45, 1.0).lerp(Color(1.0, 0.78, 0.2), clampf(ts * 1.4, 0.0, 1.0))
			if b == sel_block and h == sel_head:
				c = Color(1, 1, 1)
			fm.albedo_color = c
		# text bars: what this probe reads in the prompt (selected head in the selected block, else all heads)
		var mm: MultiMesh = d["txtbars"]
		var vals := PackedFloat32Array()
		vals.resize(T)
		var hs: Array = [sel_head] if b == sel_block else range(NH)
		for h in hs:
			var o: int = (base + h) * T
			var sc := _f32("att_txt_max.bin", base + h)
			for i in T:
				vals[i] += bin["att_txt.bin"][o + i] / 255.0 * sc
		var mx := 0.0
		for i in T:
			mx = maxf(mx, vals[i])
		for i in T:
			var hgt := 0.02 + 0.9 * vals[i] / maxf(mx, 1e-9)
			var x := -PW / 2 + (i + 0.5) * PW / T
			mm.set_instance_transform(i, Transform3D(Basis.from_scale(Vector3(PW / T * 0.8, hgt, 0.05)),
				Vector3(x, PH / 2 + 2.55 + hgt / 2, 0)))
			mm.set_instance_color(i, _token_col(i))


func _token_col(i: int) -> Color:
	var gspan: Dictionary = meta["groups"]
	for g in gspan:
		if i >= int(gspan[g][0]) and i <= int(gspan[g][1]):
			return GROUP_COLS.get(g, Color.WHITE)
	return Color(0.55, 0.48, 0.42)


func _refresh_tokens() -> void:
	## the prompt lights up with what the probe reads in the selected block (selected head)
	var base := ((step * B + sel_block) * P + probe) * NH + sel_head
	var o := base * T
	for i in T:
		var v: float = bin["att_txt.bin"][o + i] / 255.0
		var l: Label3D = token_labels[i]
		var c := _token_col(i)
		l.modulate = c.darkened(0.6).lerp(c.lightened(0.3), v)
		l.font_size = int(54 + 40 * v)


func _refresh_overlay() -> void:
	var tex: ImageTexture = null
	var gain := 1.0
	var tint := Color(1, 0.6, 0.2)
	overlay_mat.set_shader_parameter("mode", 4)
	if overlay >= 1 and overlay <= 5:
		tex = _tex("words.bin", (step * 5 + overlay - 1) * 4032, 84, 48)
		gain = 1.6
		tint = GROUP_COLS[meta["word_groups"][overlay - 1]]
	elif overlay == 6:
		overlay_mat.set_shader_parameter("mode", 5)
		tex = _labels_tex(false)
	elif overlay == 7:
		tex = _labels_tex(true)
		tint = Color(0.3, 0.55, 1.0)
	elif overlay == 8 and picked_patch >= 0:
		tex = _patch_att_tex(step)
	overlay_mat.set_shader_parameter("data", tex)
	overlay_mat.set_shader_parameter("opacity", 0.0 if tex == null else 0.8)
	overlay_mat.set_shader_parameter("gain", gain)
	overlay_mat.set_shader_parameter("tint", Vector3(tint.r, tint.g, tint.b))
	for s in S:
		var o: ShaderMaterial = tower_over[s]
		var t2: ImageTexture = null
		o.set_shader_parameter("mode", 4)
		if overlay >= 1 and overlay <= 5:
			t2 = _tex("words.bin", (s * 5 + overlay - 1) * 4032, 84, 48)
		elif overlay == 6:
			o.set_shader_parameter("mode", 5)
			t2 = _labels_tex(false)
		elif overlay == 7:
			t2 = _labels_tex(true)
		elif overlay == 8 and picked_patch >= 0:
			t2 = _patch_att_tex(s)
		o.set_shader_parameter("data", t2)
		o.set_shader_parameter("opacity", 0.0 if t2 == null else 0.75)
		o.set_shader_parameter("gain", gain)
		o.set_shader_parameter("tint", Vector3(tint.r, tint.g, tint.b))


func _labels_tex(background_only: bool) -> ImageTexture:
	var lab: PackedByteArray = bin["labels.bin"]
	var out := PackedByteArray()
	out.resize(4032)
	for i in 4032:
		out[i] = (255 if lab[i] == 0 else 0) if background_only else lab[i] * 30
	return ImageTexture.create_from_image(Image.create_from_data(84, 48, false, Image.FORMAT_L8, out))


func _patch_att_tex(s: int) -> ImageTexture:
	## where one clicked patch looks (mean over heads, blocks 4/10/16/22), its 32 strongest 2x2 cells
	var img := Image.create_empty(42, 24, false, Image.FORMAT_L8)
	var o := (s * 4032 + picked_patch) * 32
	for k in 32:
		var cell: int = bin["patchatt_idx.bin"].decode_u16((o + k) * 2)
		var v: float = bin["patchatt_val.bin"][o + k] / 255.0
		img.set_pixel(cell % 42, cell / 42, Color(v, 0, 0))
	return ImageTexture.create_from_image(img)


func _to_plane(px: float, py: float) -> Vector3:
	return Vector3((px / IMG_W - 0.5) * PW, (0.5 - py / IMG_H) * PH, 0)


func _patch_to_plane(r: float, c: float) -> Vector3:
	return Vector3(((c + 0.5) / 84.0 - 0.5) * PW, (0.5 - (r + 0.5) / 48.0) * PH, 0)


func _kp(s: int, key: String, j: int) -> Vector3:
	## a keypoint of step s, or Vector3.INF when SDPose was not confident (blurry early steps)
	var tr = meta["track"][s]
	if tr == null:
		return Vector3.INF
	var a: Array = tr[key]
	if float(a[j * 3 + 2]) < 0.2:
		return Vector3.INF
	return _to_plane(float(a[j * 3]), float(a[j * 3 + 1]))


func _refresh_tracking() -> void:
	track_mesh.clear_surfaces()
	var pts := []                       # [pos, colour]
	track_mesh.surface_begin(Mesh.PRIMITIVE_LINES)
	var nl := 0
	var bones := [[2, 3, "elbows"], [3, 4, "wrists"], [5, 6, "elbows"], [6, 7, "wrists"], [2, 5, "shoulders"], [1, 0, "face"]]
	if show["skeleton"]:
		for bn in bones:
			var a := _kp(step, "pose_keypoints_2d", bn[0])
			var c := _kp(step, "pose_keypoints_2d", bn[1])
			if a != Vector3.INF and c != Vector3.INF:
				var col: Color = JOINTS[bn[2]][1]
				track_mesh.surface_set_color(col)
				track_mesh.surface_add_vertex(a)
				track_mesh.surface_set_color(col)
				track_mesh.surface_add_vertex(c)
				nl += 1
	for jn in ["shoulders", "elbows", "wrists", "face"]:
		if show[jn]:
			for j in JOINTS[jn][0]:
				var p := _kp(step, "pose_keypoints_2d", j)
				if p != Vector3.INF:
					pts.append([p, JOINTS[jn][1]])
	if show["hands"]:
		for hk in ["hand_right_keypoints_2d", "hand_left_keypoints_2d"]:
			for j in 21:
				var p := _kp(step, hk, j)
				if p != Vector3.INF:
					pts.append([p, Color(0.5, 1.0, 0.9)])
			for f in 5:                    # the five finger chains from the wrist
				var prev := _kp(step, hk, 0)
				for k in 4:
					var cur := _kp(step, hk, 1 + f * 4 + k)
					if prev != Vector3.INF and cur != Vector3.INF:
						track_mesh.surface_set_color(Color(0.5, 1.0, 0.9))
						track_mesh.surface_add_vertex(prev)
						track_mesh.surface_set_color(Color(0.5, 1.0, 0.9))
						track_mesh.surface_add_vertex(cur)
						nl += 1
					prev = cur
	for g in [["book", "book"], ["hands word", "hands"]]:
		if show[g[0]]:
			var cc = meta["word_centroid"][g[1]][step]
			if cc != null:
				pts.append([_patch_to_plane(float(cc[0]), float(cc[1])), GROUP_COLS[g[1]]])
	if nl == 0:                             # an empty surface is an error; keep one invisible segment
		track_mesh.surface_set_color(Color(0, 0, 0, 0))
		track_mesh.surface_add_vertex(Vector3(0, 0, -1))
		track_mesh.surface_add_vertex(Vector3(0, 0, -1))
	track_mesh.surface_end()
	track_points.instance_count = pts.size()
	for i in pts.size():
		var big: float = 2.2 if pts[i][1] == GROUP_COLS["book"] or pts[i][1] == GROUP_COLS["hands"] else 1.0
		track_points.set_instance_transform(i, Transform3D(Basis.from_scale(Vector3.ONE * big), pts[i][0]))
		track_points.set_instance_color(i, pts[i][1])
	# probe marker: a ring where the selected probe sits
	probe_mesh.clear_surfaces()
	probe_mesh.surface_begin(Mesh.PRIMITIVE_LINES)
	var pr: Array = meta["probes"][probe]
	var pc := _patch_to_plane(float(pr[1]), float(pr[2]))
	var n := 24
	for k in n:
		var a0 := TAU * k / n
		var a1 := TAU * (k + 1) / n
		var col := Color(1, 1, 1) if show["probe"] else Color(0, 0, 0, 0)
		probe_mesh.surface_set_color(col)
		probe_mesh.surface_add_vertex(pc + Vector3(cos(a0), sin(a0), 0) * 0.22)
		probe_mesh.surface_set_color(col)
		probe_mesh.surface_add_vertex(pc + Vector3(cos(a1), sin(a1), 0) * 0.22)
	probe_mesh.surface_end()
	_refresh_tower_tracks()


func _refresh_tower_tracks() -> void:
	for c in tower_root.get_children():
		if c.name.begins_with("trk"):
			c.queue_free()
	var im := ImmediateMesh.new()
	im.surface_begin(Mesh.PRIMITIVE_LINES)
	var nseg := 0
	var series := []
	for jn in ["shoulders", "elbows", "wrists"]:
		if show[jn]:
			for j in JOINTS[jn][0]:
				series.append([jn, j, JOINTS[jn][1]])
	for s in S:
		var z := -s * 2.6 + 0.06
		# per-step skeleton on each tower plane
		for bn in [[2, 3], [3, 4], [5, 6], [6, 7], [2, 5]]:
			var a := _kp(s, "pose_keypoints_2d", bn[0])
			var c := _kp(s, "pose_keypoints_2d", bn[1])
			if show["skeleton"] and a != Vector3.INF and c != Vector3.INF:
				im.surface_set_color(Color(1, 0.95, 0.8, 0.8))
				im.surface_add_vertex(a + Vector3(0, 0, z))
				im.surface_set_color(Color(1, 0.95, 0.8, 0.8))
				im.surface_add_vertex(c + Vector3(0, 0, z))
				nseg += 1
	if show["trajectories"]:
		for se in series:
			var prev := Vector3.INF
			for s in S:
				var p := _kp(s, "pose_keypoints_2d", se[1])
				if p != Vector3.INF:
					p.z = -s * 2.6 + 0.06
					if prev != Vector3.INF:
						im.surface_set_color(se[2])
						im.surface_add_vertex(prev)
						im.surface_set_color(se[2])
						im.surface_add_vertex(p)
						nseg += 1
					prev = p
		for g in [["book", "book"], ["hands word", "hands"]]:
			if show[g[0]]:
				var prev := Vector3.INF
				for s in S:
					var cc = meta["word_centroid"][g[1]][s]
					if cc == null:
						continue
					var p := _patch_to_plane(float(cc[0]), float(cc[1]))
					p.z = -s * 2.6 + 0.06
					if prev != Vector3.INF:
						im.surface_set_color(GROUP_COLS[g[1]])
						im.surface_add_vertex(prev)
						im.surface_set_color(GROUP_COLS[g[1]])
						im.surface_add_vertex(p)
						nseg += 1
					prev = p
	if nseg == 0:
		im.surface_set_color(Color(0, 0, 0, 0))
		im.surface_add_vertex(Vector3(0, 0, 50))
		im.surface_add_vertex(Vector3(0, 0, 50))
	im.surface_end()
	if show["trajectories"]:
		for se in series:
			var prev := Vector3.INF
			for s in S:
				var p := _kp(s, "pose_keypoints_2d", se[1])
				if p != Vector3.INF:
					p.z = -s * 2.6 + 0.08
					_ball(p, se[2])
					if prev != Vector3.INF:
						_rod(prev, p, se[2])
					prev = p
		for g in [["book", "book"], ["hands word", "hands"]]:
			if show[g[0]]:
				var prev := Vector3.INF
				for s in S:
					var cc = meta["word_centroid"][g[1]][s]
					if cc == null:
						continue
					var p := _patch_to_plane(float(cc[0]), float(cc[1]))
					p.z = -s * 2.6 + 0.08
					_ball(p, GROUP_COLS[g[1]])
					if prev != Vector3.INF:
						_rod(prev, p, GROUP_COLS[g[1]])
					prev = p
	var mi := MeshInstance3D.new()
	mi.name = "trk"
	mi.mesh = im
	var lm := StandardMaterial3D.new()
	lm.vertex_color_use_as_albedo = true
	lm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mi.material_override = lm
	tower_root.add_child(mi)


func _glow_mat(c: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = c
	m.emission_enabled = true
	m.emission = c
	m.emission_energy_multiplier = 1.5
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	return m


func _rod(a: Vector3, b: Vector3, c: Color) -> void:
	var mi := MeshInstance3D.new()
	mi.name = "trk_rod"
	var cm := CylinderMesh.new()
	cm.top_radius = 0.06
	cm.bottom_radius = 0.06
	cm.height = a.distance_to(b)
	cm.radial_segments = 6
	mi.mesh = cm
	mi.material_override = _glow_mat(c)
	tower_root.add_child(mi)
	var mid := (a + b) / 2.0
	var dir := (b - a).normalized()
	var up := Vector3.UP
	var axis := up.cross(dir)
	var basis := Basis()
	if axis.length() > 1e-5:
		basis = Basis(axis.normalized(), acos(clampf(up.dot(dir), -1.0, 1.0)))
	mi.transform = Transform3D(basis, mid)


func _ball(p: Vector3, c: Color) -> void:
	var mi := MeshInstance3D.new()
	mi.name = "trk_ball"
	var sm := SphereMesh.new()
	sm.radius = 0.16
	sm.height = 0.32
	mi.mesh = sm
	mi.material_override = _glow_mat(c)
	mi.position = p
	tower_root.add_child(mi)


func _apply_3d() -> void:
	var lift := COL_H + 0.12 if show3d else 0.05
	for d in blocks:
		(d["panel_mi"] as Node3D).visible = not show3d
		(d["cells"]["mi"] as Node3D).visible = show3d
		(d["wall_flat"] as Node3D).visible = not show3d
		(d["wall3d"]["mi"] as Node3D).visible = show3d
		for tc in d["tile_cells"]:
			(tc["mi"] as Node3D).visible = show3d
		for n in d["track"]:
			(n as Node3D).position.z = lift
		(d["overlay_mi"] as Node3D).position.z = lift - 0.04


func _apply_layout() -> void:
	for b in B:
		var root: Node3D = blocks[b]["root"]
		root.position = Vector3(HALL_X0, 0, -(b + 1) * FLOW_DZ) if flow else Vector3(HALL_X0 + b * BLOCK_DX, 0, 0)
	noise_root.position = Vector3(HALL_X0, 0, 0) if flow else Vector3(15.0, 0, 0)
	out_root.position = Vector3(HALL_X0, 0, -(B + 2) * FLOW_DZ) if flow else Vector3(out_x, 0, 0)
	hall_rail.visible = not flow
	flow_rail.visible = flow
	play_x = 0.0
	_refresh_lines()


## where a patch sits in the world, on block b (or a station root), at the top of the columns
func _wpos(root: Node3D, r: float, c: float, top := true) -> Vector3:
	var p := root.position + _patch_to_plane(r, c)
	p.z += (COL_H * 0.75 + 0.1 if show3d else 0.08) if top else 0.08
	return p


func _tube(st: SurfaceTool, a: Vector3, b: Vector3, r: float, c: Color) -> void:
	var dv := b - a
	if dv.length() < 1e-4:
		return
	var dn := dv.normalized()
	var u := dn.cross(Vector3.UP)
	if u.length() < 0.1:
		u = dn.cross(Vector3.RIGHT)
	u = u.normalized()
	var v := dn.cross(u)
	for k in 6:
		var a0 := TAU * k / 6.0
		var a1 := TAU * (k + 1) / 6.0
		var o0 := (u * cos(a0) + v * sin(a0)) * r
		var o1 := (u * cos(a1) + v * sin(a1)) * r
		for q in [a + o0, b + o0, b + o1, a + o0, b + o1, a + o1]:
			st.set_color(c)
			st.add_vertex(q)


## a line of influence: straight between layers, an arc lifting out of the plane within one
func _link(st: SurfaceTool, a: Vector3, b: Vector3, w: float, c: Color, arc: bool, thick := 1.0) -> void:
	var r := (0.012 + 0.05 * w) * thick
	if not arc:
		_tube(st, a, b, r, c)
		return
	var ctrl := (a + b) / 2.0 + Vector3(0, 0, 0.6 + a.distance_to(b) * 0.35)
	var prev := a
	for k in range(1, 11):
		var t := k / 10.0
		var q := a.lerp(ctrl, t).lerp(ctrl.lerp(b, t), t)
		_tube(st, prev, q, r, c)
		prev = q


func _heat_col(w: float) -> Color:
	return Color(0.9, 0.25, 0.15).lerp(Color(1.0, 0.95, 0.6), clampf(w, 0.0, 1.0))


func _refresh_lines() -> void:
	if lines_menu:
		lines_menu.select(lines_mode)
	for c in lines_root.get_children():
		c.queue_free()
	if lines_mode == 0 or blocks.is_empty():
		return
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := 0
	var head := sel_head if view == 6 else -1
	var todo: Array = range(1, B) if flow else [sel_block]
	if lines_mode == 1:
		# token -> token: the probe's patch in block b reads these patches (from block b-1's output) and these words
		var pr: Array = meta["probes"][probe]
		for b in todo:
			var root: Node3D = blocks[b]["root"]
			var src_root: Node3D = blocks[b - 1]["root"] if flow else root
			var vals := _probe_cells(b, head)
			var order := range(252)
			order.sort_custom(func(x, y): return vals[x] > vals[y])
			var mx: float = vals[order[0]]
			var a := _wpos(root, float(pr[1]), float(pr[2]))
			for k in 8:
				var ci: int = order[k]
				var w: float = vals[ci] / maxf(mx, 1e-9)
				var tgt := _wpos(src_root, (ci / 21) * 4 + 1.5, (ci % 21) * 4 + 1.5)
				_link(st, a, tgt, w, _heat_col(w), not flow)
				n += 1
			# words: top 3 text tokens -> the token bars above the head rack
			var base: int = ((step * B + b) * P + probe) * NH
			var tv := PackedFloat32Array()
			tv.resize(T)
			for hh in (range(NH) if head < 0 else [head]):
				var sc := _f32("att_txt_max.bin", base + hh)
				for i in T:
					tv[i] += bin["att_txt.bin"][(base + hh) * T + i] / 255.0 * sc
			var to := range(T)
			to.sort_custom(func(x, y): return tv[x] > tv[y])
			for k in 3:
				var i: int = to[k]
				var bar := root.position + Vector3(-PW / 2 + (i + 0.5) * PW / T, PH / 2 + 2.55, 0.05)
				_link(st, a, bar, tv[i] / maxf(tv[to[0]], 1e-9), _token_col(i), true)
				n += 1
	elif lines_mode == 2:
		# body chain: how much each body probe reads each other one (x its share if attention were uniform)
		var probes: Array = meta["probes"]
		for b in todo:
			var root: Node3D = blocks[b]["root"]
			var src_root: Node3D = blocks[b - 1]["root"] if flow else root
			var links := []
			for i in BODY_PROBES:
				var vals := _probe_cells(b, head, i)
				var tot := 0.0
				for x in vals:
					tot += x
				for j in BODY_PROBES:
					if i == j:
						continue
					var cj := (int(probes[j][1]) / 4) * 21 + int(probes[j][2]) / 4
					links.append([vals[cj] / maxf(tot, 1e-9) * 252.0, i, j])
			links.sort_custom(func(x, y): return x[0] > y[0])
			var mx: float = links[0][0]
			for k in 12:
				var l: Array = links[k]
				var w: float = l[0] / maxf(mx, 1e-9)
				var a := _wpos(root, float(probes[l[1]][1]), float(probes[l[1]][2]))
				var tgt := _wpos(src_root, float(probes[l[2]][1]), float(probes[l[2]][2]))
				_link(st, a, tgt, w, _heat_col(w), not flow)
				n += 1
	elif lines_mode == 3:
		# joints from SDPose at this step; link strength = how much one joint's patch attends to the other's
		var joints := []
		var names := []
		for j in [2, 3, 4, 5, 6, 7, 0]:
			var p := _kp_px(step, "pose_keypoints_2d", j)
			if p.x >= 0:
				joints.append(p)
				names.append(j)
		for hk in ["hand_right_keypoints_2d", "hand_left_keypoints_2d"]:
			for j in [4, 8, 12, 20]:                         # thumb, index, middle, little fingertips
				var p := _kp_px(step, hk, j)
				if p.x >= 0:
					joints.append(p)
					names.append(100 + j)
		var bc = meta["word_centroid"]["book"][step]
		if bc != null:
			joints.append(Vector2(float(bc[1]), float(bc[0])))
			names.append(-1)
		var targets: Array = [out_root] + ([] if flow else [blocks[sel_block]["root"]])
		var links := []
		for i in joints.size():
			var pi: Vector2 = joints[i]
			var q := int(pi.y) * 84 + int(pi.x)
			var o := (step * 4032 + q) * 32
			for j in joints.size():
				if i == j:
					continue
				var pj: Vector2 = joints[j]
				var cj := (int(pj.y) / 2) * 42 + int(pj.x) / 2
				for k in 32:
					if bin["patchatt_idx.bin"].decode_u16((o + k) * 2) == cj:
						links.append([bin["patchatt_val.bin"][o + k] / 255.0, i, j])
						break
		links.sort_custom(func(x, y): return x[0] > y[0])
		for tr in targets:
			for k in mini(links.size(), 40):
				var l: Array = links[k]
				var a: Vector2 = joints[l[1]]
				var c2: Vector2 = joints[l[2]]
				_link(st, _wpos(tr, a.y, a.x, false), _wpos(tr, c2.y, c2.x, false), l[0], _heat_col(l[0]), true, 1.8)
				n += 1
			for jp in joints:                         # a ball on every joint the lines start from
				var c3 := _wpos(tr, jp.y, jp.x, false)
				for k in 3:
					var ax: Vector3 = [Vector3(0.09, 0, 0), Vector3(0, 0.09, 0), Vector3(0, 0, 0.09)][k]
					_tube(st, c3 - ax, c3 + ax, 0.07, Color(1, 1, 1))
	if n == 0:
		return
	var mi := MeshInstance3D.new()
	mi.mesh = st.commit()
	var m := StandardMaterial3D.new()
	m.vertex_color_use_as_albedo = true
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	mi.material_override = m
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	lines_root.add_child(mi)


## a keypoint in patch units (x = column, y = row), or (-1, -1) when SDPose was not confident
func _kp_px(s: int, key: String, j: int) -> Vector2:
	var tr = meta["track"][s]
	if tr == null:
		return Vector2(-1, -1)
	var a: Array = tr[key]
	if float(a[j * 3 + 2]) < 0.2:
		return Vector2(-1, -1)
	return Vector2(clampf(float(a[j * 3]) / 16.0, 0, 83.9), clampf(float(a[j * 3 + 1]) / 16.0, 0, 47.9))


# ------------------------------------------------------------------------------------------------ info

func _refresh_info() -> void:
	if info == null:
		return
	var b := sel_block
	var s := step
	var pr: Array = meta["probes"][probe]
	var t := "[b]Step %d · Block %d · probe: %s[/b]\n" % [s + 1, b, pr[0]]
	var hs: Array = meta["head_text_share"][s][b]
	var hd: Array = meta["head_dist"][s][b]
	var gi: Array = meta["gate_img"][s][b]
	var base := ((s * B + b) * P + probe) * NH + sel_head
	t += "\n[b]Head %d[/b] (shares key/value head %d with heads %d–%d)\n" % [sel_head, sel_head / 4,
		sel_head / 4 * 4, sel_head / 4 * 4 + 3]
	t += "• over all patches: %.0f%% of its attention goes to the prompt\n" % (float(hs[sel_head]) * 100.0)
	t += "• looks on average %.1f patches away (small = local, large = global)\n" % float(hd[sel_head])
	t += "• output gate open %.2f on the picture (0 = closed, 1 = open)\n" % float(gi[sel_head])
	t += "• for %s: %.0f%% on the prompt, gate %.2f\n" % [pr[0], _f32("att_txt_sum.bin", base) * 100.0,
		bin["gate_probe.bin"][base] / 255.0]
	# the words this head reads hardest for the probe
	var o := base * T
	var order := range(T)
	order.sort_custom(func(x, y): return bin["att_txt.bin"][o + x] > bin["att_txt.bin"][o + y])
	var words := []
	for k in 5:
		words.append("“%s”" % String(meta["tokens"][order[k]]).strip_edges())
	t += "• reads: " + ", ".join(words) + "\n"
	var wq: Array = meta["headnorm_wq"][b]
	var wo: Array = meta["headnorm_wo"][b]
	t += "• weights: |W_q| %.1f, |W_o| %.1f\n" % [float(wq[sel_head]), float(wo[sel_head])]
	# keys and locks
	var r := float(meta["keylock_r"][s][b])
	var fa := float(meta["focus_attn"][s][b])
	var fm := float(meta["focus_mlp"][s][b])
	t += "\n[b]Keys & locks (your idea)[/b]\n• MLP writes where attention wrote: r = %+.2f\n" % r
	t += "• focus (share written on the top 10%% of patches): attention %.0f%%, MLP %.0f%%  %s\n" % [fa * 100,
		fm * 100, "→ MLP sharper (lens)" if fm > fa else "→ MLP not sharper"]
	var mod: Array = meta["mod"][s][b]
	t += "• adaLN gates at this noise level: attention %.2f, MLP %.2f\n" % [float(mod[2]), float(mod[5])]
	# neurons
	var top: Array = meta["neuron_top"][s][b]
	var pn: Array = meta["probe_neurons"][s][b][probe]
	t += "\n[b]MLP[/b] most place-selective neurons: %s\n" % _ints(top.slice(0, 6))
	t += "• strongest at %s: %s\n" % [pr[0], _ints(pn.slice(0, 5))]
	if sel_neuron >= 0:
		t += "• selected neuron #%d (map shown in view 'Selected neuron')\n" % int(top[sel_neuron])
	if picked_patch >= 0:
		t += "\n[b]Clicked patch[/b] row %d, col %d (overlay 'Clicked patch attention')\n" % [picked_patch / 84, picked_patch % 84]
	info.text = t


func _ints(a: Array) -> String:
	var out := []
	for x in a:
		out.append("#%d" % int(x))
	return ", ".join(out)


# ------------------------------------------------------------------------------------------------ UI

func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	ui_layer = layer
	var panel := PanelContainer.new()
	panel.position = Vector2(10, 10)
	panel.custom_minimum_size = Vector2(330, 0)
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.1, 0.06, 0.035, 0.88)
	sb.set_corner_radius_all(8)
	sb.content_margin_left = 10
	sb.content_margin_right = 10
	sb.content_margin_top = 8
	sb.content_margin_bottom = 8
	panel.add_theme_stylebox_override("panel", sb)
	layer.add_child(panel)
	var v := VBoxContainer.new()
	panel.add_child(v)
	var title := Label.new()
	title.text = "KREA STUDIO · one real generation"
	title.add_theme_color_override("font_color", Color(1, 0.8, 0.5))
	v.add_child(title)
	step_label = Label.new()
	v.add_child(step_label)
	step_slider = HSlider.new()
	step_slider.min_value = 1
	step_slider.max_value = S
	step_slider.value = 1
	step_slider.value_changed.connect(func(x): _set_step(int(x) - 1))
	v.add_child(step_slider)
	var hb := HBoxContainer.new()
	v.add_child(hb)
	var play := Button.new()
	play.text = "▶ Play the generation"
	play.pressed.connect(func(): playing = not playing; play.text = "❚❚ Pause" if playing else "▶ Play the generation")
	hb.add_child(play)
	var spd := HSlider.new()
	spd.min_value = 0.25
	spd.max_value = 4
	spd.step = 0.25
	spd.value = 1
	spd.custom_minimum_size = Vector2(100, 0)
	spd.value_changed.connect(func(x): speed = x)
	hb.add_child(spd)
	v.add_child(_caption("Probe point (whose attention the heads show)"))
	var po := OptionButton.new()
	for pr in meta["probes"]:
		po.add_item(pr[0])
	probe_menu = po
	po.item_selected.connect(func(i): probe = i; _refresh_heads(); _refresh_blocks(); _refresh_tokens(); _refresh_tracking(); _refresh_lines(); _refresh_info())
	v.add_child(po)
	v.add_child(_caption("Block panels show"))
	var vo := OptionButton.new()
	for name in VIEWS:
		vo.add_item(name)
	view_menu = vo
	vo.item_selected.connect(func(i): view = i; _refresh_blocks(); _refresh_lines())
	v.add_child(vo)
	v.add_child(_caption("Overlay on every picture"))
	var oo := OptionButton.new()
	for name in OVERLAYS:
		oo.add_item(name)
	overlay_menu = oo
	oo.item_selected.connect(func(i): overlay = i; _refresh_overlay())
	v.add_child(oo)
	v.add_child(_caption("Influence lines (token to token)"))
	var lo := OptionButton.new()
	for name in LINE_MODES:
		lo.add_item(name)
	lo.select(lines_mode)
	lines_menu = lo
	lo.item_selected.connect(func(i): lines_mode = i; _refresh_lines())
	v.add_child(lo)
	var h3 := HBoxContainer.new()
	v.add_child(h3)
	var c3 := CheckBox.new()
	c3.text = "3D columns"
	c3.button_pressed = show3d
	c3.toggled.connect(func(on): show3d = on; _apply_3d(); _refresh_lines())
	h3.add_child(c3)
	var cf := CheckBox.new()
	cf.text = "Flow: layer to layer"
	cf.button_pressed = flow
	cf.toggled.connect(func(on): flow = on; _apply_layout(); _go(Vector3(-2, 0, 0)) if on else _go(Vector3(-1, 0, 0)))
	h3.add_child(cf)
	flow_box = cf
	v.add_child(_caption("Tracking"))
	var grid := GridContainer.new()
	grid.columns = 2
	v.add_child(grid)
	for k in show:
		var cb := CheckBox.new()
		cb.text = k
		cb.button_pressed = show[k]
		cb.toggled.connect(func(on): show[k] = on; _refresh_tracking())
		grid.add_child(cb)
	v.add_child(_caption("Go to"))
	var g2 := GridContainer.new()
	g2.columns = 3
	v.add_child(g2)
	for item in [["Prompt", Vector3(0, 0, 0)], ["Noise", Vector3(15, 0, 0)], ["Hall / Flow", Vector3(-1, 0, 0)],
			["Preview", Vector3(out_x, 0, 0)], ["Picture", Vector3(out_x + 16, 0, 0)], ["Tower", Vector3(tower_x, 0, 0)]]:
		var btn := Button.new()
		btn.text = item[0]
		var p: Vector3 = item[1]
		btn.pressed.connect(func(): _go(p))
		g2.add_child(btn)
	block_slider = HSlider.new()
	block_slider.min_value = 0
	block_slider.max_value = B - 1
	block_slider.value = sel_block
	block_slider.value_changed.connect(func(x): _select_block(int(x), true))
	v.add_child(_caption("Block (fly to it)"))
	v.add_child(block_slider)
	v.add_child(_caption("Drag: look · WASD: fly · Alt+drag: orbit · click: pick a head, neuron, weight or patch · Space: play · ←/→: step"))
	# info on the right
	var ip := PanelContainer.new()
	ip.add_theme_stylebox_override("panel", sb)
	ip.anchor_left = 1.0
	ip.anchor_right = 1.0
	ip.offset_left = -400
	ip.offset_right = -10
	ip.offset_top = 10
	ip.offset_bottom = 520
	layer.add_child(ip)
	info = RichTextLabel.new()
	info.bbcode_enabled = true
	info.fit_content = true
	info.custom_minimum_size = Vector2(380, 0)
	info.add_theme_font_size_override("normal_font_size", 14)
	info.add_theme_font_size_override("bold_font_size", 15)
	ip.add_child(info)
	step_label.text = ""


func _caption(t: String) -> Label:
	var l := Label.new()
	l.text = t
	l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	l.custom_minimum_size = Vector2(310, 0)
	l.add_theme_font_size_override("font_size", 12)
	l.add_theme_color_override("font_color", Color(0.85, 0.72, 0.55))
	return l


func _go(p: Vector3) -> void:
	if flow and p.x < 0:                           # the stack, seen from the side so the layers spread out
		var zc := -(B + 2) * FLOW_DZ / 2.0
		cam.fly_to(Vector3(HALL_X0 + 30, 12, zc + 34), Vector3(HALL_X0, 0, zc + 6))
		return
	if flow and absf(p.x - 15) < 1:
		p = noise_root.position
	elif flow and absf(p.x - out_x) < 1:
		p = out_root.position
	if p.x < 0:                                    # the whole hall
		var c := (HALL_X0 + out_x) / 2.0
		cam.fly_to(Vector3(c, 40, 230), Vector3(c, 0, 0))
	elif absf(p.x - tower_x) < 1:
		cam.fly_to(Vector3(tower_x + 11, 2.5, 6), Vector3(tower_x + 0.5, 0.8, -9))
	else:
		cam.fly_to(p + Vector3(0, 0.8, 11), p + Vector3(0, 0.8, 0))


func _select_block(b: int, fly := false) -> void:
	sel_block = b
	sel_neuron = -1
	if fly:
		var rp: Vector3 = blocks[b]["root"].position
		if flow:
			cam.fly_to(rp + Vector3(9, 3, 10), rp + Vector3(0, 0.5, -2))
		else:
			cam.fly_to(rp + Vector3(0, 0.5, 14), rp + Vector3(0, 0.5, 0))
	_refresh_blocks()
	_refresh_heads()
	_refresh_tokens()
	_refresh_lines()
	_refresh_info()
	_sync_menus()


func _set_step(s: int) -> void:
	step = clampi(s, 0, S - 1)
	if step_slider and int(step_slider.value) != step + 1:
		step_slider.set_value_no_signal(step + 1)
	_refresh_all()


# ------------------------------------------------------------------------------------------------ input

func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseButton and e.button_index == MOUSE_BUTTON_LEFT:
		if e.pressed:
			press_pos = e.position
		elif e.position.distance_to(press_pos) < 5.0:
			_click(e.position)
	elif e is InputEventKey and e.pressed and not e.echo:
		match e.keycode:
			KEY_SPACE:
				playing = not playing
			KEY_RIGHT:
				_set_step(step + 1)
			KEY_LEFT:
				_set_step(step - 1)


func _click(sp: Vector2) -> void:
	var o := cam.project_ray_origin(sp)
	var d := cam.project_ray_normal(sp)
	var best := {}
	var best_t := INF
	for pk in pickables:
		if absf(d.z) < 1e-6 or not (pk["root"] as Node3D).visible:
			continue
		var t: float = (_pick_z(pk) - o.z) / d.z
		if t <= 0 or t >= best_t:
			continue
		var hit := o + d * t
		var r := _pick_rect(pk)
		if r.has_point(Vector2(hit.x, hit.y)):
			best = pk
			best_t = t
			best["hit"] = hit
	if best.is_empty():
		return
	var dd: Dictionary = best["data"]
	var hit: Vector3 = best["hit"]
	var r2 := _pick_rect(best)
	var u := (hit.x - r2.position.x) / r2.size.x
	var v := 1.0 - (hit.y - r2.position.y) / r2.size.y
	match best["kind"]:
		"head":
			sel_block = dd["b"]
			sel_head = dd["h"]
			view = 6 if view != 6 else view
			_select_block(sel_block)
		"panel", "plane":
			if dd.has("b"):
				sel_block = dd["b"]
			var row := clampi(int(v * 48), 0, 47)
			var col := clampi(int(u * 84), 0, 83)
			picked_patch = row * 84 + col
			overlay = 8
			for i in P:                            # clicking near a probe point selects it
				var pr: Array = meta["probes"][i]
				if absi(int(pr[1]) - row) <= 1 and absi(int(pr[2]) - col) <= 1:
					probe = i
			_refresh_all()
		"wall":
			var b: int = dd["b"]
			var n := clampi(int(v * 128), 0, 127) * 128 + clampi(int(u * 128), 0, 127)
			var top: Array = meta["neuron_top"][step][b]
			var bestk := -1
			var bestd := 1e9
			for k in 16:                           # snap to the nearest place-selective neuron
				var tn := int(top[k])
				var dist := Vector2(tn % 128 - n % 128, tn / 128 - n / 128).length()
				if dist < bestd:
					bestd = dist
					bestk = k
			sel_block = b
			sel_neuron = bestk
			view = 7
			_refresh_blocks()
			_refresh_info()
			_sync_menus()
		"weight":
			sel_block = dd["b"]
			_refresh_info()
			info.text += "\n[b]Weight %s, block %d[/b]: mean |w| %.4f (64×64 thumbnail of the real matrix)" % [
				meta["weight_mats"][dd["k"]], dd["b"], float(meta["weight_mean"][dd["b"]][dd["k"]])]


func _process(dt: float) -> void:
	if playing:
		# the pulse rides the rail from the prompt to the preview; arriving there is one denoising step
		play_x += dt * speed * (60.0 if not flow else 14.0)
		var end := out_x if not flow else (B + 2) * FLOW_DZ
		if play_x > end:
			play_x = 0.0
			_set_step((step + 1) % S)
		var cur := 0
		if flow:
			pulse.position = Vector3(HALL_X0, -PH / 2 - 0.3, -play_x)
			cur = int(play_x / FLOW_DZ + 0.5) - 1
		else:
			pulse.position = Vector3(play_x, 0.0, -1.5)
			cur = int((play_x - HALL_X0) / BLOCK_DX + 0.5)
		for b in B:
			var m: ShaderMaterial = blocks[b]["panel"]
			m.set_shader_parameter("glow", 0.6 if b == cur else 0.0)
			(blocks[b]["cells"]["mat"] as ShaderMaterial).set_shader_parameter("glow", 0.8 if b == cur else 0.0)


# ------------------------------------------------------------------------------------------------ animations

## --anim=<dir>: record frame sequences for the README GIFs, then quit
func _auto_anim(dir: String) -> void:
	DirAccess.make_dir_recursive_absolute(dir)
	await get_tree().create_timer(1.0).timeout
	cam.smoothing = 1000.0                       # the camera follows the script exactly while recording
	ui_layer.visible = false                     # clean frames for the GIFs
	# 1. the picture emerges in the residual stream of block 18, step by step, as 3D columns
	view = 0
	lines_mode = 2
	sel_block = 18
	var rp: Vector3 = blocks[18]["root"].position
	cam.fly_to(rp + Vector3(5, 2.5, 10), rp + Vector3(0, 0.5, 0))
	var f := 0
	for s in S:
		_set_step(s)
		for k in 6:
			await _grab(dir, "steps", f)
			f += 1
	# 2. keys & locks: orbit block 22
	view = 2
	lines_mode = 1
	sel_block = 22
	_set_step(5)
	rp = blocks[22]["root"].position
	f = 0
	for k in 60:
		var a := -0.9 + 1.8 * k / 59.0
		cam.fly_to(rp + Vector3(sin(a) * 11, 3, cos(a) * 11), rp + Vector3(0, 0.8, 0))
		await _grab(dir, "orbit", f)
		f += 1
	# 3. flow: the residual stream travels layer to layer
	flow = true
	flow_box.set_pressed_no_signal(true)
	_apply_layout()
	view = 0
	lines_mode = 2
	_refresh_all()
	f = 0
	for k in 90:
		var z := -k * (B + 1) * FLOW_DZ / 89.0
		cam.fly_to(Vector3(HALL_X0 + 14, 6, z + 14), Vector3(HALL_X0, 0, z - 4))
		pulse.position = Vector3(HALL_X0, -PH / 2 - 0.3, z - 4)
		await _grab(dir, "flow", f)
		f += 1
	flow = false
	flow_box.set_pressed_no_signal(false)
	_apply_layout()
	# 4. the preview with the joints and their links, step by step
	lines_mode = 3
	overlay = 6
	f = 0
	cam.fly_to(Vector3(out_x, 1.0, 6.5), Vector3(out_x, 0.8, 0))
	for s in S:
		_set_step(s)
		for k in 6:
			await _grab(dir, "joints", f)
			f += 1
	get_tree().quit()


func _grab(dir: String, name: String, f: int) -> void:
	await get_tree().process_frame
	await get_tree().process_frame
	var img := get_viewport().get_texture().get_image()
	img.resize(800, 450, Image.INTERPOLATE_BILINEAR)
	img.save_jpg("%s/%s_%03d.jpg" % [dir, name, f], 0.9)


# ------------------------------------------------------------------------------------------------ verification

## --shot=<dir>: fly to a few stations, save screenshots, quit. Lets the build be looked at without a person.
func _auto_shot(dir: String) -> void:
	DirAccess.make_dir_recursive_absolute(dir)
	var plan := [
		["hall", func(): _go(Vector3(-1, 0, 0))],
		["block", func(): view = 0; _set_step(5); _select_block(14, true)],
		["heads", func(): cam.fly_to(Vector3(HALL_X0 + 14 * BLOCK_DX, 4.2, 7), Vector3(HALL_X0 + 14 * BLOCK_DX, 3.6, 0))],
		["keylock", func(): view = 2; _refresh_blocks(); _select_block(22, true)],
		["prompt", func(): _go(Vector3(0, 0, 0))],
		["preview", func(): overlay = 3; _set_step(5); _refresh_overlay(); _go(Vector3(out_x, 0, 0))],
		["tower", func(): overlay = 0; _refresh_overlay(); _go(Vector3(tower_x, 0, 0))],
		["cells3d", func(): overlay = 0; _refresh_overlay(); view = 2; lines_mode = 1; _set_step(5); var rp: Vector3 = blocks[22]["root"].position; sel_block = 22; _refresh_lines(); cam.fly_to(rp + Vector3(7, 5, 8), rp + Vector3(0, 0.5, 0))],
		["chain", func(): view = 0; lines_mode = 2; sel_block = 18; _refresh_blocks(); _refresh_lines(); var rp: Vector3 = blocks[18]["root"].position; cam.fly_to(rp + Vector3(-4, -3, 9), rp + Vector3(0, 1, 0))],
		["flow", func(): flow = true; flow_box.set_pressed_no_signal(true); _apply_layout(); lines_mode = 2; view = 0; _refresh_blocks(); _refresh_lines(); _go(Vector3(-2, 0, 0))],
		["flowtok", func(): lines_mode = 1; _refresh_lines(); _select_block(12, true)],
		["joints", func(): flow = false; flow_box.set_pressed_no_signal(false); _apply_layout(); lines_mode = 3; overlay = 0; _refresh_overlay(); _set_step(7); cam.fly_to(Vector3(out_x, 1.0, 5.5), Vector3(out_x, 1.0, 0))],
		["heads3d", func(): view = 0; lines_mode = 0; _set_step(5); var rp: Vector3 = blocks[14]["root"].position; cam.fly_to(rp + Vector3(2.5, 6.5, 5), rp + Vector3(0, 3.6, 0))],
		["mlp_wall", func(): var rp: Vector3 = blocks[14]["root"].position; cam.fly_to(rp + Vector3(1, -7.5, 5), rp + Vector3(-1.5, -4.6, 0))],
		["probe_attention", func(): view = 5; probe = 1; lines_mode = 1; _refresh_all(); var rp: Vector3 = blocks[20]["root"].position; sel_block = 20; _refresh_lines(); cam.fly_to(rp + Vector3(-6, 3, 9), rp + Vector3(0, 0.5, 0))],
		["word_overlay", func(): view = 8; overlay = 4; lines_mode = 0; _refresh_all(); var rp: Vector3 = blocks[10]["root"].position; cam.fly_to(rp + Vector3(0, 0.5, 8), rp)],
		["hall_3d", func(): view = 0; overlay = 0; probe = 0; _refresh_all(); cam.fly_to(Vector3(HALL_X0 + 60, 14, 40), Vector3(HALL_X0 + 100, 0, 0))],
		["hands", func(): overlay = 6; _refresh_overlay(); _set_step(7); cam.fly_to(Vector3(out_x, 1.2, 3.2), Vector3(out_x, 1.2, 0))],
	]
	for item in plan:
		item[1].call()
		await get_tree().create_timer(1.6).timeout
		get_viewport().get_texture().get_image().save_png("%s/%s.png" % [dir, item[0]])
	get_tree().quit()
