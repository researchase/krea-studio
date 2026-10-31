shader_type spatial;
render_mode cull_back;

// Data as columns. One MultiMesh of unit boxes per surface; every instance finds its own cell from INSTANCE_ID and
// reads its value from the same data texture the flat views use, so a step change still only swaps a texture.
//   mode 0: picture (colour = the step's guess, height = darkness)   mode 1: heat
//   mode 2: diverging (128 = zero; height = |value|)                  mode 3: PCA colours (height = distance from mean)
//   mode 4: keys & locks RGB (height = the larger write)

uniform sampler2D data : filter_nearest, repeat_disable;
uniform sampler2D pic : source_color, filter_linear, repeat_disable;
uniform int mode = 1;
uniform int grid_w = 84;
uniform int grid_h = 48;
uniform vec2 size = vec2(8.4, 4.8);
uniform float height = 1.0;
uniform float gain = 1.0;
uniform float glow = 0.0;
uniform float fill = 0.86;

varying vec3 col;

vec3 heat(float v) {
	v = clamp(v, 0.0, 1.0);
	vec3 a = vec3(0.07, 0.04, 0.03);
	vec3 b = vec3(0.55, 0.10, 0.08);
	vec3 c = vec3(0.95, 0.45, 0.10);
	vec3 d = vec3(1.0, 0.85, 0.45);
	vec3 e = vec3(1.0, 1.0, 0.92);
	if (v < 0.25) return mix(a, b, v / 0.25);
	if (v < 0.5) return mix(b, c, (v - 0.25) / 0.25);
	if (v < 0.8) return mix(c, d, (v - 0.5) / 0.3);
	return mix(d, e, (v - 0.8) / 0.2);
}

void vertex() {
	int id = INSTANCE_ID;
	int c = id % grid_w;
	int r = id / grid_w;
	vec2 uv = (vec2(float(c), float(r)) + 0.5) / vec2(float(grid_w), float(grid_h));
	vec4 t = texture(data, uv);
	float v;
	if (mode == 0) {
		col = texture(pic, uv).rgb;
		v = 1.0 - dot(col, vec3(0.3, 0.55, 0.15));
	} else if (mode == 1) {
		v = clamp(t.r * gain, 0.0, 1.0);
		col = heat(pow(v, 0.8));
	} else if (mode == 2) {
		float s = (t.r - 0.5) * 2.0 * gain;
		v = abs(s);
		col = s > 0.0 ? mix(vec3(0.09, 0.06, 0.05), vec3(1.0, 0.62, 0.2), clamp(s, 0.0, 1.0))
		              : mix(vec3(0.09, 0.06, 0.05), vec3(0.25, 0.55, 1.0), clamp(-s, 0.0, 1.0));
	} else if (mode == 3) {
		col = clamp((t.rgb - 0.5) * gain + 0.5, 0.0, 1.0);
		v = length(t.rgb - 0.5) * 1.5;
	} else {
		col = t.rgb;
		v = max(t.r, t.b);
	}
	vec2 cell = size / vec2(float(grid_w), float(grid_h));
	float cx = (float(c) + 0.5) * cell.x - size.x * 0.5;
	float cy = size.y * 0.5 - (float(r) + 0.5) * cell.y;
	float h = max(clamp(v, 0.0, 1.2) * height, 0.004);
	VERTEX = vec3(VERTEX.x * cell.x * fill + cx, VERTEX.y * cell.y * fill + cy, (VERTEX.z + 0.5) * h);
}

void fragment() {
	ALBEDO = col;
	EMISSION = col * (0.35 + glow);
	ROUGHNESS = 0.6;
}
