#version 400
#extension GL_ARB_separate_shader_objects: enable
#extension GL_ARB_shading_language_420pack: enable

layout (location = 0) in vec4 color;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 pos;
layout (location = 3) in vec3 light_dir;

layout (location = 0) out vec4 frag_color;

void main() {
    float diff = max(dot(normal, light_dir), 0.0);
    frag_color = vec4(diff * color.rgb, 1.0);
}
