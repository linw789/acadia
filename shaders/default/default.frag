#version 450

layout (binding = 0) uniform FrameData {
    mat4 pers_view_matrix;
    vec4 camera_lookat;
} frame_data;

layout (location = 0) in vec4 color;
layout (location = 1) in vec3 normal;

layout (location = 0) out vec4 frag_color;

void main() {
    vec3 light_dir = frame_data.camera_lookat.xyz;
    float diff = max(dot(normal, light_dir), 0.0);
    frag_color = vec4(diff * color.rgb, 1.0);
}
