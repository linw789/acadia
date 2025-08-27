use crate::assets::{MeshId, TextureId, ShaderId};

#[derive(Default)]
pub struct Entity {
    pub mesh_id: MeshId,
    pub texture_id: TextureId,
    pub shader_id: ShaderId,
}

impl Entity {
    pub fn add_mesh(&mut self, id: MeshId) {
        self.mesh_id = id;
    }

    pub fn add_shader(&mut self, id: ShaderId) {
        self.shader_id = id;
    }

    pub fn add_texture(&mut self, id: TextureId) {
        self.texture_id = id;
    }
}
