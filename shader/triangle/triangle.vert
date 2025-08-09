#version 400
#extension GL_ARB_separate_shader_objects: enable
#extension GL_ARB_shading_language_420pack: enable

layout (binding = 0) uniform MvpTransform {
    mat4 view_mat;
    mat4 pers_mat;
} transform;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 color;

layout (location = 0) out vec4 out_color;

void main() {
    out_color = color;
    gl_Position = transform.pers_mat * transform.view_mat * pos;
    gl_Position.y = -gl_Position.y;
}
