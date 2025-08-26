use crate::{gui::font::FontBitmap, assets::{MeshId, TextureId}};
use ash::{Device, util::read_spv, vk};
use std::{fs::File, path::Path};

#[derive(Default)]
pub struct Entity {
    pub mesh: MeshId,
    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,
    pub texture: TextureId,
    pub font_bitmap: FontBitmap,
}

impl Entity {
    pub fn add_mesh(&mut self, id: MeshId) {
        self.mesh = id;
    }

    pub fn add_vertex_shader<P: AsRef<Path>>(&mut self, device: &Device, vert_spv_path: P) {
        let mut spv_file = File::open(vert_spv_path).unwrap();
        let spv = read_spv(&mut spv_file).unwrap();
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&spv);
        self.vertex_shader = unsafe {
            device
                .create_shader_module(&vertex_shader_info, None)
                .unwrap()
        };
    }

    pub fn add_fragment_shader<P: AsRef<Path>>(&mut self, device: &Device, frag_spv_path: P) {
        let mut spv_file = File::open(frag_spv_path).unwrap();
        let spv = read_spv(&mut spv_file).unwrap();
        let fragment_shader_info = vk::ShaderModuleCreateInfo::default().code(&spv);
        self.fragment_shader = unsafe {
            device
                .create_shader_module(&fragment_shader_info, None)
                .unwrap()
        };
    }

    pub fn add_texture(&mut self, id: TextureId) {
        self.texture = id;
    }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.vertex_shader, None);
            device.destroy_shader_module(self.fragment_shader, None);
        }
        self.vertex_shader = vk::ShaderModule::null();
        self.fragment_shader = vk::ShaderModule::null();
    }
}
