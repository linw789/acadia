use ::ash::{Device, Instance, khr::swapchain, vk};

pub struct Swapchain {
    loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    present_images: Vec<vk::Image>,
    surface_extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        inst: &Instance,
        device: &Device,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        surface_capabilities: &vk::SurfaceCapabilitiesKHR,
        present_mode: vk::PresentModeKHR,
        window_size: vk::Extent2D,
    ) -> Self {
        // 0 means there is no limit on max image count.
        let max_image_count = if surface_capabilities.max_image_count == 0 {
            u32::MAX
        } else {
            surface_capabilities.max_image_count
        };

        let desired_image_count =
            u32::min(surface_capabilities.min_image_count + 1, max_image_count);

        let surface_extent = match surface_capabilities.current_extent.width {
            u32::MAX => window_size,
            _ => surface_capabilities.current_extent,
        };

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let loader = swapchain::Device::new(&inst, &device);
        let swapchain_createinfo = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe {
            loader
                .create_swapchain(&swapchain_createinfo, None)
                .unwrap()
        };

        let present_images = unsafe { loader.get_swapchain_images(swapchain).unwrap() };

        let present_image_views: Vec<vk::ImageView> = present_images
            .iter()
            .map(|&image| {
                let view_createinfo = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);

                unsafe {
                    device
                        .create_image_view(&view_createinfo, None)
                        .expect("Failed to create image view.")
                }
            })
            .collect();

        Self {
            loader,
            swapchain,
            present_images,
            surface_extent,
        }
    }

    pub fn acquire_next_image(&self, acquire_sema: vk::Semaphore) -> u32 {
        let (present_image_index, _is_suboptimal) = unsafe {
            self.loader
                .acquire_next_image(self.swapchain, u64::MAX, acquire_sema, vk::Fence::null())
                .unwrap()
        };
        // TODO: warn if sub-optimal
        present_image_index
    }

    pub fn queue_present(
        &self,
        present_queue: vk::Queue,
        wait_sema: vk::Semaphore,
        present_index: u32,
    ) {
        let wait_semaphores = [wait_sema];
        let swapchains = [self.swapchain];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.loader
                .queue_present(present_queue, &present_info)
                .unwrap();
        }
    }

    pub fn present_images(&self) -> &Vec<vk::Image> {
        &self.present_images
    }

    pub fn surface_extent(&self) -> vk::Extent2D {
        self.surface_extent
    }

    pub fn destroy(&self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
