class_name OrbitCamera
extends Camera3D

## Desktop camera: orbit, pan, zoom. Stands in for three.js OrbitControls, which the web
## build used and which has no Godot equivalent.
##
## Only ever active outside a session. In VR the headset owns the view, and the machine
## is moved instead of the camera -- you carry the thing you are looking at rather than
## flying around it, which is the whole difference between the two.

@export var target := Vector3.ZERO
@export var distance := 108.0
@export var min_distance := 2.0
@export var max_distance := 900.0

## radians. pitch is clamped short of the poles, where orbiting turns into gimbal spin
@export var yaw := 0.0
@export var pitch := -0.2
const PITCH_LIMIT := 1.45

@export var orbit_speed := 0.006
@export var pan_speed := 0.0016
@export var zoom_step := 1.12

## damping. The original had enableDamping on, and without it the camera feels like it is
## snapping rather than being pushed.
@export var smoothing := 18.0

var _want_yaw := 0.0
var _want_pitch := 0.0
var _want_dist := 0.0
var _want_target := Vector3.ZERO

var _orbiting := false
var _panning := false
var _looking := false          ## RMB held: Unreal-style free look + fly

## editor-style flight. Speed is PROPORTIONAL TO DISTANCE from the orbit target -- close
## to a cell you creep, out at lane scale you sweep -- which is what makes one key feel
## right at every zoom. The wheel (while RMB is down) scales it; Shift triples it.
var fly_scale := 0.85
var _fly_vel := Vector3.ZERO

## While something else owns the drag -- painting on the input plane -- motion must not
## also orbit. Button events still pass, so press/release state cannot get stuck.
var blocked := false


func _ready() -> void:
	_want_yaw = yaw
	_want_pitch = pitch
	_want_dist = distance
	_want_target = target
	_apply()


## Point at something and back off far enough to see all of it.
##
## Fits each axis against its own field of view rather than putting a sphere around the
## box. For a shape like this machine -- 140 long, 21 tall -- the bounding sphere is
## nearly all empty space, and framing by its radius stands you about 50% too far back.
func frame(box: AABB, margin := 1.12) -> void:
	_want_target = box.get_center()
	var vfov := deg_to_rad(fov) * 0.5
	var aspect := 1.0
	var vp := get_viewport()
	if vp:
		var s := vp.get_visible_rect().size
		if s.y > 0.0:
			aspect = s.x / s.y
	var hfov := atan(tan(vfov) * aspect)
	var half := box.size * 0.5 * margin
	# plus half the depth, or the near face of a deep object is already behind the camera
	var d := maxf(half.x / tan(hfov), half.y / tan(vfov)) + half.z
	_want_dist = clampf(d, min_distance, max_distance)


## A step's camera move: eye and target, converted into the orbit parameters so the
## existing damping IS the flight -- no second animation system, and grabbing the mouse
## mid-flight just takes over, the way OrbitControls did.
func fly_to(eye: Vector3, target: Vector3) -> void:
	_want_target = target
	var v := eye - target
	_want_dist = clampf(v.length(), min_distance, max_distance)
	_want_yaw = atan2(v.x, v.z)
	_want_pitch = clampf(asin(v.y / maxf(v.length(), 0.001)), -PITCH_LIMIT, PITCH_LIMIT)


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseButton:
		match e.button_index:
			MOUSE_BUTTON_LEFT:
				_orbiting = e.pressed
			MOUSE_BUTTON_RIGHT:
				_looking = e.pressed
				Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if e.pressed \
					else Input.MOUSE_MODE_VISIBLE
			MOUSE_BUTTON_MIDDLE:
				_panning = e.pressed
			MOUSE_BUTTON_WHEEL_UP:
				if e.pressed:
					if _looking:
						fly_scale = minf(fly_scale * 1.2, 6.0)   # Unreal: wheel tunes speed
					else:
						_want_dist = maxf(min_distance, _want_dist / zoom_step)
			MOUSE_BUTTON_WHEEL_DOWN:
				if e.pressed:
					if _looking:
						fly_scale = maxf(fly_scale / 1.2, 0.1)
					else:
						_want_dist = minf(max_distance, _want_dist * zoom_step)
	elif e is InputEventMouseMotion:
		if blocked:
			return
		# First person is the default grammar: dragging with EITHER button turns your
		# head from a fixed eye -- the network stays where it stands, you look around,
		# then WASD walks you where you are now facing. The old orbit read as the whole
		# landscape swinging past, which is exactly what Ashwin reported.
		#
		# Deliberate orbiting is still here: Alt+left-drag circles the target, for
		# walking around one cube on purpose. Shift+left-drag and middle-drag pan.
		if _panning or (_orbiting and e.shift_pressed):
			# pan in the camera's own plane, scaled by distance so it feels the same
			# whether you are inspecting one cube or looking at the whole lane
			var right := global_transform.basis.x
			var up := global_transform.basis.y
			_want_target -= (right * e.relative.x - up * e.relative.y) * pan_speed * _want_dist
		elif _orbiting and e.alt_pressed:
			_want_yaw -= e.relative.x * orbit_speed
			_want_pitch = clampf(_want_pitch - e.relative.y * orbit_speed, -PITCH_LIMIT, PITCH_LIMIT)
		elif _looking or _orbiting:
			# look pivots at the CAMERA: the target swings around to stay ahead, so
			# flight and any later orbit remain one coherent state. Vertical is
			# inverted-from-FPS-default at Ashwin's request: drag up looks down,
			# the way grabbing the view and pushing it feels.
			var pos := position
			_want_yaw -= e.relative.x * LOOK_SPEED
			_want_pitch = clampf(_want_pitch + e.relative.y * LOOK_SPEED, -PITCH_LIMIT, PITCH_LIMIT)
			yaw = _want_yaw
			pitch = _want_pitch
			var dir := Vector3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
			target = pos - dir * distance
			_want_target = target
			_apply()


const LOOK_SPEED := 0.0034


func _process(dt: float) -> void:
	_fly(dt)
	var k := 1.0 - exp(-smoothing * dt)      # frame-rate independent damping
	yaw = lerp_angle(yaw, _want_yaw, k)
	pitch = lerpf(pitch, _want_pitch, k)
	distance = lerpf(distance, _want_dist, k)
	target = target.lerp(_want_target, k)
	_apply()


## WASD flies the camera where it looks -- W is the whole of zooming: go closer to the
## thing. Q and E sink and rise, Shift hurries. Works with or without the right button
## held; the two-handed Unreal grip and the lazy one-handed cruise are both real.
##
## Velocity is EASED both ways -- keys ramp it up, letting go coasts it down -- because
## an instant start and a dead stop are precisely what "the control feels off" means.
func _fly(dt: float) -> void:
	if not current or blocked:
		_fly_vel = Vector3.ZERO
		return
	var f := (1.0 if Input.is_key_pressed(KEY_W) else 0.0) \
		- (1.0 if Input.is_key_pressed(KEY_S) else 0.0)
	var r := (1.0 if Input.is_key_pressed(KEY_D) else 0.0) \
		- (1.0 if Input.is_key_pressed(KEY_A) else 0.0)
	var u := (1.0 if Input.is_key_pressed(KEY_E) else 0.0) \
		- (1.0 if Input.is_key_pressed(KEY_Q) else 0.0)
	var b := global_transform.basis
	var wish := -b.z * f + b.x * r + Vector3.UP * u
	if wish.length() > 1.0:
		wish = wish.normalized()
	# creep at cell scale, sweep at lane scale: speed follows the orbit distance
	var speed := clampf(_want_dist * fly_scale, 6.0, 320.0) \
		* (3.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0)
	_fly_vel = _fly_vel.lerp(wish * speed, 1.0 - exp(-7.0 * dt))
	if _fly_vel.length_squared() < 0.01:
		_fly_vel = Vector3.ZERO
		return
	# carry the whole orbit state along: the target rides ahead of the camera, so
	# letting go of the keys leaves a normal orbit around wherever you arrived --
	# flying at something IS the zoom, and there is no radius bookkeeping to fight
	_want_target += _fly_vel * dt
	target += _fly_vel * dt
	_apply()


func _apply() -> void:
	var dir := Vector3(
		cos(pitch) * sin(yaw),
		sin(pitch),
		cos(pitch) * cos(yaw))
	position = target + dir * distance
	look_at(target, Vector3.UP)
