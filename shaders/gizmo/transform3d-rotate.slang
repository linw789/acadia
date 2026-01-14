struct PerFrame {
    float4x4 pers_view_model_matrix;
}

[[vk::binding(0, 0)]]
ConstantBuffer<PerFrame> per_frame;

struct VertexInput {
    float3 position;
    float3 color;
}

struct FragmentInput {
    float3 color;
}

struct VertexOutput {
    FragmentInput frag_input: FragmentInput;
    float4 position: SV_POSITION;
}

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;
    output.position = mul(per_frame.pers_view_model_matrix, float4(input.position, 1.0));
    output.frag_input.color = input.color;
    return output;
}

struct Fragment {
    float4 color;
}

[shader("fragment")]
Fragment fragment_main(FragmentInput frag_input: FragmentInput) {
    Fragment fragment = {};
    fragment.color = float4(frag_input.color, 1.0);
    return fragment;
}
