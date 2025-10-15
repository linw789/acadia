struct AssembledVertex {
    float2 position;
    float2 uv;
}

struct FragmentInput {
    float2 uv;
}

struct VertexOutput {
    FragmentInput frag_input: FragmentInput; 
    float4 position: SV_POSITION;
}

[shader("vertex")]
VertexOutput vertex_main(AssembledVertex assembled) {
    VertexOutput output;
    output.frag_input.uv = assembled.uv;
    output.position = float4(assembled.position, 0.0, 1.0);
    return output;
}

// [[vk::binding(0, 1)]]
// Sampler2D shadow_depth_sampler;

struct Fragment {
    float4 color;
}

[shader("fragment")]
Fragment fragment_main(FragmentInput frag_input: FragmentInput) {
    // float4 texel = shadow_depth_sampler.Sample(frag_input.uv);

    Fragment fragment;
    fragment.color = float4(frag_input.uv, 0.0, 1.0);
    return fragment;
}
