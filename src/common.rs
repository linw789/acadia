#[derive(Clone, Debug, Default, Copy)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub color: [f32; 4],
    pub normal: [f32; 4],
    pub uv: [f32; 2],
}
