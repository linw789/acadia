
struct AssembledVertex {
    float2 position;
    float4 color;
    float2 uv;
}

struct FragmentInput {
    float2 uv;
}

struct VertexOutput {
    FragmentInput fragment_input: FragmentInput;
    float4 position: SV_POSITION;
}

[shader("vertex")]
VertexOutput vertex_main(AssembledVertex assembled) {
    VertexOutput output;
    output.fragment_input.uv = assembled.uv;
    output.position = float4(assembled.position, 0.0, 1.0);
    return output;
}

[[vk::binding(0, 0)]]
Sampler2D font_texture;

struct Fragment {
    float4 color;
}

[shader("fragment")]
Fragment fragment_main(FragmentInput fragment_input: FragmentInput) {
    Fragment fragment;
    float4 texel = font_texture.Sample(fragment_input.uv);
    if (texel.r == 0.0) {
        discard;
    } else {
        fragment.color = texel;
    }
    return fragment;
}
