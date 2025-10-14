struct PerFrame {
    float4x4 pers_view_matrix;
    float3 light_dir;
}
[[vk::binding(0, 0)]]
ConstantBuffer<PerFrame> per_frame;

struct AssembledVertex {
    float3 position;
    float4 color;
    float3 normal;
}

struct FragmentInput {
    float4 color;
    float3 normal;
}

struct VertexOutput {
    FragmentInput frag_input: FragmentInput; 
    float4 position: SV_POSITION;
}

[shader("vertex")]
VertexOutput vertex_main(AssembledVertex assembled) {
    VertexOutput output;
    output.frag_input.color = assembled.color;
    output.frag_input.normal = assembled.normal;
    output.position = mul(per_frame.pers_view_matrix, float4(assembled.position, 1.0));
    return output;
}

// [[vk::binding(0, 1)]]
// Sampler2D tex_sampler;

struct Fragment {
    float4 color;
}

[shader("fragment")]
Fragment fragment_main(FragmentInput frag_input: FragmentInput) {
    float shading = max(dot(frag_input.normal, -per_frame.light_dir), 0.0);
    
    Fragment fragment;
    fragment.color = float4(shading * frag_input.color.rgb, 1.0);
    return fragment;
}

