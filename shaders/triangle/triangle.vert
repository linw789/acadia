#version 400
#extension GL_ARB_separate_shader_objects: enable
#extension GL_ARB_shading_language_420pack: enable

layout (binding = 0) uniform FrameData {
    mat4 pers_view_matrix;
    vec4 light_dir;

} frame_data;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 color;
layout (location = 2) in vec4 normal;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec3 out_pos;
layout (location = 3) out vec3 out_light_dir;

void main() {
    gl_Position = frame_data.pers_view_matrix * pos;

    out_color = color;
    out_normal = normal.xyz;
    out_pos = pos.xyz;
    out_light_dir = frame_data.light_dir.xyz;
}
