#version 450

layout (binding = 0) uniform FrameData {
    mat4 pers_view_matrix;
    vec4 camera_lookat;
} frame_data;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 color;
layout (location = 2) in vec4 normal;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec3 out_normal;

void main() {
    gl_Position = frame_data.pers_view_matrix * pos;

    out_color = color;
    out_normal = normal.xyz;
}
