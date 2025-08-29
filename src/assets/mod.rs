pub mod mesh;
pub mod shader;
pub mod texture;

use crate::gui::font::Font;
use ash::{Device, vk};
use mesh::Mesh;
use shader::Shader;
use std::{path::Path, vec::Vec};
use texture::{Texture, TextureIngredient, bake_textures};

#[derive(Default)]
pub struct Assets {
    texture_ingredients: Vec<TextureIngredient>,
    textures: Vec<Texture>,
    meshes: Vec<Mesh>,
    shaders: Vec<Shader>,

    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
}

pub type TextureId = u32;
pub type MeshId = u32;
pub type ShaderId = u32;

pub const TEXTURE_ID_INVALID: u32 = u32::MAX;
pub const MESH_ID_INVALID: u32 = u32::MAX;
pub const SHADER_ID_INVALID: u32 = u32::MAX;

impl Assets {
    pub fn new(device_memory_properties: vk::PhysicalDeviceMemoryProperties) -> Self {
        Self {
            texture_ingredients: Vec::new(),
            textures: Vec::new(),
            meshes: Vec::new(),
            shaders: Vec::new(),
            device_memory_properties,
        }
    }

    pub fn add_texture_ingredient(&mut self, ingredient: TextureIngredient) -> TextureId {
        self.texture_ingredients.push(ingredient);
        (self.texture_ingredients.len() - 1) as TextureId
    }

    pub fn bake_textures(&mut self, device: &Device, cmd_buf: vk::CommandBuffer, queue: vk::Queue) {
        self.textures = bake_textures(
            device,
            cmd_buf,
            queue,
            &self.device_memory_properties,
            &self.texture_ingredients,
        );
        self.texture_ingredients.clear();
    }

    pub fn add_mesh<P: AsRef<Path>>(&mut self, device: &Device, obj_file: P) -> MeshId {
        let mesh = Mesh::from_obj(device, &self.device_memory_properties, obj_file);
        self.meshes.push(mesh);
        (self.meshes.len() - 1) as MeshId
    }

    pub fn add_shader<P: AsRef<Path>>(
        &mut self,
        device: &Device,
        vert_spv: P,
        frag_spv: P,
    ) -> ShaderId {
        self.shaders.push(Shader::new(device, vert_spv, frag_spv));
        (self.shaders.len() - 1) as ShaderId
    }

    pub fn texture<'a>(&'a self, id: TextureId) -> &'a Texture {
        &self.textures[id as usize]
    }

    pub fn mesh<'a>(&'a self, id: MeshId) -> &'a Mesh {
        &self.meshes[id as usize]
    }

    pub fn shader<'a>(&'a self, id: ShaderId) -> &'a Shader {
        &self.shaders[id as usize]
    }

    pub fn destruct(&mut self, device: &Device) {
        for tex in self.textures.iter_mut() {
            tex.destruct(device);
        }
        for mesh in self.meshes.iter_mut() {
            mesh.destruct(device);
        }
        for shader in self.shaders.iter_mut() {
            shader.destruct(device);
        }
        self.textures.clear();
        self.meshes.clear();
        self.shaders.clear();
    }
}
