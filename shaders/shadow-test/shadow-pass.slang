struct PerFrame {
    float4x4 light_projection_matrix;
}
[[vk::binding(0, 0)]]
ConstantBuffer<PerFrame> per_frame;

struct AssembledVertex {
    float3 position;
}

struct VertexOutput {
    float4 position: SV_POSITION;
}

[shader("vertex")]
VertexOutput vertex_main(AssembledVertex assembled) {
    VertexOutput output;
    output.position = mul(per_frame.light_projection_matrix, float4(assembled.position, 1.0));
    return output;
}
