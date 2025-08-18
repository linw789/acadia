#version 450

layout (binding = 0) uniform FrameData {
    mat4 pers_view_matrix;
    vec4 camera_lookat;
} frame_data;

layout (location = 0) in vec3 pos;
layout (location = 1) in vec4 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 tex_coord;

layout (location = 0) out vec2 out_tex_coord;

void main() {
    gl_Position = frame_data.pers_view_matrix * vec4(pos, 1.0);
    out_tex_coord = tex_coord;
}
