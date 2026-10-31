shader_type spatial;
render_mode cull_disabled;

// The MLP wall as a surface: 16384 neurons on a 128 x 128 sheet, each vertex lifted by how strongly its neuron fires
// on average over the picture. Orange = positive, blue = negative (SwiGLU outputs can be both).

uniform sampler2D data : filter_nearest, repeat_disable;
uniform float height = 0.9;
uniform float gain = 1.0;

varying vec3 col;

void vertex() {
	float s = (texture(data, UV).r - 0.5) * 2.0 * gain;
	col = s > 0.0 ? mix(vec3(0.12, 0.08, 0.06), vec3(1.0, 0.62, 0.2), clamp(s, 0.0, 1.0))
	              : mix(vec3(0.12, 0.08, 0.06), vec3(0.25, 0.55, 1.0), clamp(-s, 0.0, 1.0));
	VERTEX.z += abs(s) * height;
}

void fragment() {
	ALBEDO = col;
	EMISSION = col * 0.3;
	ROUGHNESS = 0.7;
}
