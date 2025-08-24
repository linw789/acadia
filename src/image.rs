use crate::util::find_memorytype_index;
use ::ash::{Device, vk};

#[derive(Default)]
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
}

impl Image {
    pub fn new_depth_image(
        device: &Device,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
    ) -> Self {
        let (image, view, memory) = unsafe {
            let depth_image_createinfo = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let depth_image = device.create_image(&depth_image_createinfo, None).unwrap();
            let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index);

            let depth_image_memory = device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .expect("Unable to bind depth image memory.");

            let depth_image_view_info = vk::ImageViewCreateInfo::default()
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                )
                .image(depth_image)
                .format(depth_image_createinfo.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            (depth_image, depth_image_view, depth_image_memory)
        };

        Self {
            image,
            view,
            memory,
        }
    }

    pub fn new_texture_image(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        let image_createinfo = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&image_createinfo, None).unwrap() };

        let depth_image_memory_req = unsafe { device.get_image_memory_requirements(image) };
        let depth_image_memory_index = find_memorytype_index(
            &depth_image_memory_req,
            &memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("Unable to find suitable memory index for depth image.");

        let image_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(depth_image_memory_req.size)
            .memory_type_index(depth_image_memory_index);

        let memory = unsafe { device.allocate_memory(&image_allocate_info, None).unwrap() };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("Unable to bind depth image memory.");
        }

        let view_createinfo = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping { 
                r: vk::ComponentSwizzle::R, 
                g: vk::ComponentSwizzle::R, 
                b: vk::ComponentSwizzle::R, 
                a: vk::ComponentSwizzle::R, 
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { device.create_image_view(&view_createinfo, None).unwrap() };

        Self {
            image,
            view,
            memory,
        }
    }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }

        self.image = vk::Image::null();
        self.view = vk::ImageView::null();
        self.memory = vk::DeviceMemory::null();
    }
}
