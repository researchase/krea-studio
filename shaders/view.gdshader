shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_opaque;

// One shader for every data surface in the studio. The data arrives as raw bytes (L8 or RGB8 textures), and the
// colour ramp lives here so a step change only swaps a texture, never recolours pixels in GDScript.
//   mode 0: picture (sRGB image)          mode 1: heat (L8, warm ramp)
//   mode 2: diverging (L8, 128 = zero)    mode 3: PCA colours (RGB8, 128 = zero)
//   mode 4: mask/heat overlay, alpha by value, tinted
//   mode 5: region labels (L8 holds id * 30) with a palette

uniform sampler2D data : filter_nearest, repeat_disable;
uniform sampler2D pic : source_color, filter_linear_mipmap, repeat_disable;
uniform int mode = 1;
uniform float gain = 1.0;
uniform float opacity = 1.0;
uniform float glow = 0.0;
uniform vec3 tint = vec3(1.0, 0.6, 0.2);
uniform vec3 pal[8];

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

void fragment() {
	vec4 t = texture(data, UV);
	vec3 col;
	float a = opacity;
	if (mode == 0) {
		col = texture(pic, UV).rgb;
	} else if (mode == 1) {
		col = heat(pow(t.r * gain, 0.8));
	} else if (mode == 2) {
		float s = (t.r - 0.5) * 2.0 * gain;
		col = s > 0.0 ? mix(vec3(0.09, 0.06, 0.05), vec3(1.0, 0.62, 0.2), clamp(s, 0.0, 1.0))
		              : mix(vec3(0.09, 0.06, 0.05), vec3(0.25, 0.55, 1.0), clamp(-s, 0.0, 1.0));
	} else if (mode == 3) {
		col = clamp((t.rgb - 0.5) * gain + 0.5, 0.0, 1.0);
	} else if (mode == 4) {
		float v = clamp(t.r * gain, 0.0, 1.0);
		col = mix(tint, heat(v), 0.35);
		a = opacity * smoothstep(0.08, 0.6, v);
	} else {
		int id = int(round(t.r * 255.0 / 30.0));
		col = pal[clamp(id, 0, 7)];
		a = id == 0 ? opacity * 0.15 : opacity;
	}
	ALBEDO = col;
	EMISSION = col * glow;
	ALPHA = a;
}
