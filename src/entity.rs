use crate::mesh::Mesh;
use ash::{Device, util::read_spv, vk};
use std::{fmt::Debug, fs::File, path::Path};

#[derive(Default)]
pub struct Entity {
    pub mesh: Mesh,
    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,
}

impl Entity {
    pub fn add_mesh<P: AsRef<Path> + Debug>(&mut self, mesh_path: P) {
        self.mesh = Mesh::from_obj(mesh_path);
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

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.vertex_shader, None);
            device.destroy_shader_module(self.fragment_shader, None);
        }
        self.vertex_shader = vk::ShaderModule::null();
        self.fragment_shader = vk::ShaderModule::null();
    }
}
