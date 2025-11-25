struct PerFrame {
    float4x4 pers_view_matrix;
}

[[vk::binding(0, 0)]]
ConstantBuffer<PerFrame> per_frame;

struct VertexInput {
    float3 position;
    // instance attributes
    uint id;
    float3 color;
    float4x4 transform;
}

struct FragmentInput {
    uint id;
    float3 color;
}

struct VertexOutput {
    FragmentInput frag_input: FragmentInput; 
    float4 position: SV_POSITION;
}

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;
    output.frag_input.id = input.id;
    output.frag_input.color = input.color;

    float4 world_pos = mul(input.transform, float4(input.position, 1.0));
    output.position = mul(per_frame.pers_view_matrix, world_pos);
    return output;
}

struct Fragment {
    float4 color;
    uint id;
}

[shader("fragment")]
Fragment fragment_main(FragmentInput frag_input: FragmentInput) {
    Fragment fragment;
    fragment.color = float4(frag_input.color, 1.0);
    fragment.id = frag_input.id;
    return fragment;
}
