struct PerFrame {
    float4x4 pers_view_matrix;
}
[[vk::binding(0, 0)]]
ConstantBuffer<PerFrame> per_frame;

struct AssembledVertex {
    float3 position;
    float4 color;
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
    output.frag_input.color = assembled.color;
    output.position = mul(per_frame.pers_view_matrix, float4(assembled.position, 1.0));
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
