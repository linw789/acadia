struct PerFrame {
    float4x4 pers_view_matrix;
}
[vk::binding(0, 0)]
ConstantBuffer<PerFrame> per_frame;

struct PushConstant {
    float4x4 transform;
}
[vk::push_constant]
PushConstant push_constant;

struct AssembledVertex {
    float3 position;
}

struct FragmentInput {
    float4 color;
}

struct VertexOutput {
    FragmentInput frag_input: FragmentInput; 
    float4 position: SV_POSITION;
}

[shader("vertex")]
VertexOutput vertex_main(AssembledVertex assembled) {
    VertexOutput output;
    output.frag_input.color = float4(0.8, 0.8, 0.8, 1.0);
    float4 world_position = mul(push_constant.transform, float4(assembled.position, 1.0));
    output.position = mul(per_frame.pers_view_matrix, world_position);
    return output;
}

struct Fragment {
    float4 color;
}

[shader("fragment")]
Fragment fragment_main(FragmentInput frag_input: FragmentInput) {
    Fragment fragment;
    fragment.color = frag_input.color;
    return fragment;
}
