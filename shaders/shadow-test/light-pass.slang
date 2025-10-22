struct PerFrame {
    float4x4 pers_view_matrix;
    float4x4 light_projection;
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
    float3 position;
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
    output.frag_input.position = assembled.position;
    output.position = mul(per_frame.pers_view_matrix, float4(assembled.position, 1.0));
    return output;
}

[[vk::binding(1, 0)]]
Sampler2D<float> shadow_depth_sampler;

struct Fragment {
    float4 color;
}

[shader("fragment")]
Fragment fragment_main(FragmentInput frag_input: FragmentInput) {
    float shading = max(dot(frag_input.normal, -per_frame.light_dir), 0.0);
    float4 pos_light_space = mul(per_frame.light_projection, float4(frag_input.position, 1.0));
    float2 shadow_depth_uv = pos_light_space.xy / pos_light_space.w;
    float max_depth = shadow_depth_sampler.Sample(shadow_depth_uv);
    float shadow = pos_light_space.z >= max_depth;
    
    Fragment fragment;
    fragment.color = float4(shading * frag_input.color.rgb * shadow, 1.0);
    return fragment;
}

