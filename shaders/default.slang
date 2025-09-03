struct PerFrame {
    float4x4 pers_view_matrix;
    float4 camera_lookat;
}
[[vk::binding(0, 0)]]
ConstantBuffer<PerFrame> per_frame;

struct AssembledVertex {
    float3 position;
    float4 color;
    float3 normal;
    float2 uv;
}

struct FragmentInput {
    float4 color;
    float3 normal;
    float2 uv;
}

struct VertexOutput {
    FragmentInput fragment_input: FragmentInput; 
    float4 position: SV_POSITION;
}

[shader("vertex")]
VertexOutput vertex_main(AssembledVertex assembled) {
    VertexOutput output;
    output.fragment_input.color = assembled.color;
    output.fragment_input.normal = assembled.normal;
    output.fragment_input.uv = assembled.uv;
    output.position = mul(per_frame.pers_view_matrix, float4(assembled.position, 1.0));
    return output;
}

[[vk::binding(0, 1)]]
Sampler2D tex_sampler;

struct Fragment {
    float4 color;
}

[shader("fragment")]
Fragment fragment_main(FragmentInput fragment_input: FragmentInput) {
    float3 light_dir = per_frame.camera_lookat.xyz;
    float diff = max(dot(fragment_input.normal, light_dir), 0.0);

    float4 texel = tex_sampler.Sample(fragment_input.uv);

    Fragment fragment;
    fragment.color = float4(diff * texel.rgb, 1.0);
    return fragment;
}
