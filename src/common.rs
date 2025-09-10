#[derive(Clone, Debug, Default, Copy)]
#[repr(C, packed)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, Default, Copy)]
#[repr(C, packed)]
pub struct Vertex2D {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}
