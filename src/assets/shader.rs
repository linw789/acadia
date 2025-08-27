use ash::{Device, util::read_spv, vk};
use std::{fs::File, path::Path};

pub struct Shader {
    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,
}

impl Shader {
    pub(super) fn new<P: AsRef<Path>>(device: &Device, vert_spv: P, frag_spv: P) -> Self {
        let mut spv_file = File::open(vert_spv).unwrap();
        let spv = read_spv(&mut spv_file).unwrap();
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&spv);
        let vertex_shader = unsafe {
            device
                .create_shader_module(&vertex_shader_info, None)
                .unwrap()
        };

        let mut spv_file = File::open(frag_spv).unwrap();
        let spv = read_spv(&mut spv_file).unwrap();
        let fragment_shader_info = vk::ShaderModuleCreateInfo::default().code(&spv);
        let fragment_shader = unsafe {
            device
                .create_shader_module(&fragment_shader_info, None)
                .unwrap()
        };

        Self {
            vertex_shader,
            fragment_shader,
        }
    }

    pub(super) fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.vertex_shader, None);
            device.destroy_shader_module(self.fragment_shader, None);
        }
        self.vertex_shader = vk::ShaderModule::null();
        self.fragment_shader = vk::ShaderModule::null();
    }
}
