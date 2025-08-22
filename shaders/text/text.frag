#version 450

layout(set = 0, binding = 0) uniform sampler2D tex_sampler;

layout(location = 0) in vec2 uv;

layout (location = 0) out vec4 frag_color;

void main() {
    vec4 texel = texture(tex_sampler, uv);
    if (texel.r == 0.0) {
        discard;
    } else {
        frag_color = vec4(texel.r, 0.0, 0.0, 1.0);
    }
}
