use ::ash::{Device, Instance, khr::swapchain, vk};

pub struct Swapchain {
    loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    present_images: Vec<vk::Image>,
    image_extent: vk::Extent2D,
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
        let loader = swapchain::Device::new(&inst, &device);

        let mut sc = Self {
            loader,
            swapchain: vk::SwapchainKHR::null(),
            present_images: Vec::new(),
            image_extent: vk::Extent2D::default(),
        };

        let _ = sc.create(
            surface,
            surface_format,
            surface_capabilities,
            present_mode,
            window_size,
        );

        sc
    }

    /// Re-create
    pub fn recreate(
        &mut self,
        device: &Device,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        surface_capabilities: &vk::SurfaceCapabilitiesKHR,
        present_mode: vk::PresentModeKHR,
        window_size: vk::Extent2D,
    ) -> bool {
        let old_swapchain = self.create(
            surface,
            surface_format,
            surface_capabilities,
            present_mode,
            window_size,
        );

        if let Some(old) = old_swapchain {
            unsafe {
                device.device_wait_idle().unwrap();
                self.loader.destroy_swapchain(old, None);
            }
        }

        old_swapchain.is_some()
    }

    /// Create a swapchain if non exists. Create a new swapchain is the window_size doesn't match the one
    /// in the existing swapchain, and return the vk-handle to the old swapchain.
    fn create(
        &mut self,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        surface_capabilities: &vk::SurfaceCapabilitiesKHR,
        present_mode: vk::PresentModeKHR,
        image_extent: vk::Extent2D,
    ) -> Option<vk::SwapchainKHR> {
        if self.image_extent == image_extent {
            return None;
        }

        let old_swapchain = if self.swapchain == vk::SwapchainKHR::null() {
            vk::SwapchainKHR::null()
        } else {
            self.swapchain
        };

        // 0 means there is no limit on max image count.
        let max_image_count = if surface_capabilities.max_image_count == 0 {
            u32::MAX
        } else {
            surface_capabilities.max_image_count
        };

        let desired_image_count =
            u32::min(surface_capabilities.min_image_count + 1, max_image_count);

        self.image_extent = match surface_capabilities.current_extent.width {
            u32::MAX => image_extent,
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

        let swapchain_createinfo = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(image_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .old_swapchain(old_swapchain);

        self.swapchain = unsafe {
            self.loader
                .create_swapchain(&swapchain_createinfo, None)
                .unwrap()
        };

        self.present_images = unsafe { self.loader.get_swapchain_images(self.swapchain).unwrap() };

        if old_swapchain == vk::SwapchainKHR::null() {
            None
        } else {
            Some(old_swapchain)
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

    pub fn image_extent(&self) -> vk::Extent2D {
        self.image_extent
    }

    pub fn destroy(&self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
