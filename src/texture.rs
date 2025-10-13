use ash::{Device, vk};

#[derive(Default)]
pub struct Texture {
    pub image_index: u32,
    pub sampler: vk::Sampler,
}

impl Texture {
    pub fn new(device: &Device, image_index: u32) -> Self {
        let sampler_createinfo = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS);

        let sampler = unsafe { device.create_sampler(&sampler_createinfo, None).unwrap() };

        Self {
            image_index,
            sampler,
        }
    }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
        self.sampler = vk::Sampler::null();
    }
}
