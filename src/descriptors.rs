use crate::{texture::Texture, buffer::Buffer};
use ash::{Device, vk};
use std::vec::Vec;

const PER_FRAME_DATA_SET_INDEX: usize = 0;
const DEV_GUI_SAMPLER_SET_INDEX: usize = 1;
const SAMPLER_SET_INDEX_START: usize = 2;

#[derive(Default)]
pub struct Descriptors {
    pool: vk::DescriptorPool,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    sets: Vec<vk::DescriptorSet>,
}

impl Descriptors {
    pub fn new(device: &Device, max_sampler_count: usize) -> Self {
        let pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 100,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 100,
                },
            ];

            let pool_createinfo = vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .max_sets(50 as u32)
                .pool_sizes(&pool_sizes);

            unsafe {
                device
                    .create_descriptor_pool(&pool_createinfo, None)
                    .unwrap()
            }
        };

        let (set_layouts, sets) = {
            let per_frame_data_layout = {
                let desc_set_layout_bindings = [vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

                let layout_info =
                    vk::DescriptorSetLayoutCreateInfo::default().bindings(&desc_set_layout_bindings);
                unsafe {
                    device
                        .create_descriptor_set_layout(&layout_info, None)
                        .unwrap()
                }
            };

            let sampler_set_layout = {
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

            let mut set_layouts = Vec::with_capacity(1 + max_sampler_count);
            set_layouts.push(per_frame_data_layout);
            set_layouts.push(sampler_set_layout); // for dev gui
            for _ in 0..max_sampler_count {
                set_layouts.push(sampler_set_layout);
            }

            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(&set_layouts)
                .descriptor_pool(pool);

            let sets = unsafe {
                device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap()
            };

            (vec![per_frame_data_layout, sampler_set_layout], sets)
        };
        Self {
            pool,
            set_layouts,
            sets,
        }
    }

    pub fn update_per_frame_set(&self, device: &Device, buffer: &Buffer, buffer_size: u64) {
        let desc_buf_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.buf)
            .offset(0)
            .range(buffer_size);

        let desc_writes = [vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .dst_set(self.sets[0])
            .dst_binding(0)
            .dst_array_element(0)
            .buffer_info(std::slice::from_ref(&desc_buf_info))];

        unsafe {
            device.update_descriptor_sets(&desc_writes, &[]);
        }
    }

    pub fn update_dev_gui_sampler_set(&self, device: &Device, texture: &Texture) {
        let desc_image_info = vk::DescriptorImageInfo::default()
            .sampler(texture.sampler)
            .image_view(texture.image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let desc_writes = [vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_set(self.sets[DEV_GUI_SAMPLER_SET_INDEX])
            .dst_binding(0)
            .dst_array_element(0)
            .image_info(std::slice::from_ref(&desc_image_info))];

        unsafe {
            device.update_descriptor_sets(&desc_writes, &[]);
        }
    }

    pub fn update_sampler_set(&self, device: &Device, index: usize, texture: &Texture) {
        let desc_image_info = vk::DescriptorImageInfo::default()
            .sampler(texture.sampler)
            .image_view(texture.image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let desc_writes = [vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_set(self.sets[SAMPLER_SET_INDEX_START + index])
            .dst_binding(0)
            .dst_array_element(0)
            .image_info(std::slice::from_ref(&desc_image_info))];

        unsafe {
            device.update_descriptor_sets(&desc_writes, &[]);
        }
    }

    pub fn per_frame_data_set(&self) -> vk::DescriptorSet {
        self.sets[PER_FRAME_DATA_SET_INDEX]
    }

    pub fn dev_gui_sampler_set(&self) -> vk::DescriptorSet {
        self.sets[DEV_GUI_SAMPLER_SET_INDEX]
    }

    pub fn sampler_set(&self, index: usize) -> vk::DescriptorSet {
        self.sets[SAMPLER_SET_INDEX_START + index]
    }

    // pub fn update_dev_gui_set(&self, device: &Device, texture: &Texture) {
    //     let desc_image_info = vk::DescriptorImageInfo::default()
    //         .sampler(texture.sampler)
    //         .image_view(texture.image.view)
    //         .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    //     let desc_writes = [vk::WriteDescriptorSet::default()
    //         .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    //         .dst_set(self.sets[1])
    //         .dst_binding(0)
    //         .dst_array_element(0)
    //         .image_info(std::slice::from_ref(&desc_image_info))];

    //     unsafe {
    //         device.update_descriptor_sets(&desc_writes, &[]);
    //     }
    // }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            for layout in &self.set_layouts {
                device.destroy_descriptor_set_layout(*layout, None);
            }
        }
        self.sets.clear();
    }
}
