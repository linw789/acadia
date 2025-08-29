use crate::{assets::texture::Texture, buffer::Buffer};
use ash::{Device, vk};
use std::vec::Vec;

#[derive(Default)]
pub struct Descriptors {
    pool: vk::DescriptorPool,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    sets: Vec<vk::DescriptorSet>,
}

impl Descriptors {
    pub fn new(device: &Device) -> Self {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: 10,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 10,
            },
        ];

        let pool_createinfo = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(4 as u32)
            .pool_sizes(&pool_sizes);

        let pool = unsafe {
            device
                .create_descriptor_pool(&pool_createinfo, None)
                .unwrap()
        };

        let set_layouts = Self::create_set_layouts(device);

        let sets = {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(&set_layouts)
                .descriptor_pool(pool);

            unsafe {
                device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap()
            }
        };

        Self {
            pool,
            set_layouts,
            sets,
        }
    }

    pub fn update_default_set(
        &self,
        device: &Device,
        buffer: &Buffer,
        buffer_size: u64,
        texture: &Texture,
    ) {
        let desc_buf_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.buf)
            .offset(0)
            .range(buffer_size);
        let desc_image_info = vk::DescriptorImageInfo::default()
            .sampler(texture.sampler)
            .image_view(texture.image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let desc_writes = [
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .dst_set(self.sets[0])
                .dst_binding(0)
                .dst_array_element(0)
                .buffer_info(std::slice::from_ref(&desc_buf_info)),
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_set(self.sets[0])
                .dst_binding(1)
                .dst_array_element(0)
                .image_info(std::slice::from_ref(&desc_image_info)),
        ];

        unsafe {
            device.update_descriptor_sets(&desc_writes, &[]);
        }
    }

    pub fn update_dev_gui_set(&self, device: &Device, texture: &Texture) {
        let desc_image_info = vk::DescriptorImageInfo::default()
            .sampler(texture.sampler)
            .image_view(texture.image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let desc_writes = [vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_set(self.sets[1])
            .dst_binding(0)
            .dst_array_element(0)
            .image_info(std::slice::from_ref(&desc_image_info))];

        unsafe {
            device.update_descriptor_sets(&desc_writes, &[]);
        }
    }

    pub fn set_layout(&self, index: usize) -> vk::DescriptorSetLayout {
        self.set_layouts[index]
    }

    pub fn set(&self, index: usize) -> vk::DescriptorSet {
        self.sets[index]
    }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            for layout in &self.set_layouts {
                device.destroy_descriptor_set_layout(*layout, None);
            }
            device.destroy_descriptor_pool(self.pool, None);
        }
        self.set_layouts.clear();
        self.sets.clear();
    }

    fn create_set_layouts(device: &Device) -> Vec<vk::DescriptorSetLayout> {
        let default_set_layout = {
            let desc_set_layout_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];

            let layout_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&desc_set_layout_bindings);
            unsafe {
                device
                    .create_descriptor_set_layout(&layout_info, None)
                    .unwrap()
            }
        };

        let dev_gui_set_layout = {
            let desc_set_layout_bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

            let layout_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&desc_set_layout_bindings);
            unsafe {
                device
                    .create_descriptor_set_layout(&layout_info, None)
                    .unwrap()
            }
        };

        vec![default_set_layout, dev_gui_set_layout]
    }
}
