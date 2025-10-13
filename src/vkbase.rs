use crate::{image::Image, swapchain::Swapchain};
use ash::{Device, Entry, Instance, ext::debug_utils, khr, vk};
use std::{borrow::Cow, error::Error, ffi, os::raw::c_char, sync::Arc};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number;

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        || message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
    {
        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("?")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("?")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
        };

        println!(
            "[{message_severity:?}:{message_type:?}] [{message_id_name} ({message_id_number})] : {message}\n",
        );

        if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            panic!("Vulkan validation failed.");
        }
    }

    vk::FALSE
}

pub fn pick_present_image_format(surface_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    let mut format_index = 0;
    for (i, sf) in surface_formats.iter().enumerate() {
        if (sf.format == vk::Format::R8G8B8A8_UNORM) || (sf.format == vk::Format::B8G8R8A8_UNORM) {
            format_index = i;
            break;
        }
    }
    let result = surface_formats[format_index];
    assert!(
        result.format != vk::Format::UNDEFINED,
        "Failed to find a proper surface format."
    );
    result
}

fn create_present_image_views(
    device: &Device,
    images: &[vk::Image],
    surface_format: vk::SurfaceFormatKHR,
    image_views: &mut Vec<vk::ImageView>,
) {
    for image in images {
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
            .image(*image);
        let view = unsafe {
            device
                .create_image_view(&view_createinfo, None)
                .expect("Failed to create image view.")
        };
        image_views.push(view);
    }
}

pub struct VkBase {
    pub inst: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<Device>,
    pub debug_util_loader: debug_utils::Instance,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub surface_loader: khr::surface::Instance,

    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub present_queue: vk::Queue,

    surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,

    pub depth_format: vk::Format,

    pub swapchain: Swapchain,
    present_mode: vk::PresentModeKHR,
    pub present_image_views: Vec<vk::ImageView>,

    pub cmd_pool: vk::CommandPool,
}

impl VkBase {
    pub fn new(window: &Window) -> Result<VkBase, Box<dyn Error>> {
        let entry = Entry::linked();
        let inst = {
            let layer_names = [c"VK_LAYER_KHRONOS_validation"];
            let layer_names_raw: Vec<*const c_char> =
                layer_names.iter().map(|name| name.as_ptr()).collect();
            let mut extension_names =
                ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())
                    .unwrap()
                    .to_vec();
            extension_names.push(debug_utils::NAME.as_ptr());

            let appinfo = vk::ApplicationInfo::default()
                .application_name(c"Acadia")
                .application_version(0)
                .engine_name(c"Acadia Vulkan Renderer")
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 3, 0));

            let create_flags = vk::InstanceCreateFlags::default();

            let createinfo = vk::InstanceCreateInfo::default()
                .application_info(&appinfo)
                .enabled_layer_names(&layer_names_raw)
                .enabled_extension_names(&extension_names)
                .flags(create_flags);

            unsafe { entry.create_instance(&createinfo, None).unwrap() }
        };

        let (debug_util_loader, debug_messenger) = {
            let debuginfo = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_util_loader = debug_utils::Instance::new(&entry, &inst);
            let debug_messenger = unsafe {
                debug_util_loader
                    .create_debug_utils_messenger(&debuginfo, None)
                    .unwrap()
            };
            (debug_util_loader, debug_messenger)
        };

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &inst,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        };
        let surface_loader = khr::surface::Instance::new(&entry, &inst);

        let (physical_device, queue_family_index) = unsafe {
            let phy_devices = inst.enumerate_physical_devices().unwrap();
            let (physical_device, graphics_queue_family_index) = phy_devices
                .iter()
                .find_map(|physical_device| {
                    inst.get_physical_device_queue_family_properties(*physical_device)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let support_graphics_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *physical_device,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if support_graphics_and_surface {
                                Some((*physical_device, index))
                            } else {
                                None
                            }
                        })
                })
                .expect("Couldn't find suitable physical device.");

            (physical_device, graphics_queue_family_index as u32)
        };

        unsafe {
            let features = inst.get_physical_device_features(physical_device);
            assert!(features.sampler_anisotropy == 1);
        }

        let device = {
            let device_extension_names_raw = [khr::swapchain::NAME.as_ptr()];
            let features = vk::PhysicalDeviceFeatures::default()
                .shader_clip_distance(true)
                .sampler_anisotropy(true);
            let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true);

            let priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);
            let device_createinfo = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features)
                .push_next(&mut vk13_features);
            unsafe {
                Arc::new(
                    inst.create_device(physical_device, &device_createinfo, None)
                        .unwrap(),
                )
            }
        };

        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let surface_format = pick_present_image_format(&surface_formats);

        let depth_format = vk::Format::D32_SFLOAT;

        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let present_mode = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap()
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO)
        };

        let swapchain = Swapchain::new(
            &inst,
            &device,
            surface,
            surface_format,
            &surface_capabilities,
            present_mode,
            vk::Extent2D {
                width: window.inner_size().width,
                height: window.inner_size().height,
            },
        );

        let mut present_image_views = Vec::with_capacity(swapchain.present_images().len());
        create_present_image_views(
            &device,
            swapchain.present_images(),
            surface_format,
            &mut present_image_views,
        );

        let cmd_pool = unsafe {
            let pool_createinfo = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);
            device.create_command_pool(&pool_createinfo, None).unwrap()
        };

        let device_memory_properties =
            unsafe { inst.get_physical_device_memory_properties(physical_device) };

        Ok(Self {
            inst,
            physical_device,
            device,
            debug_util_loader,
            debug_messenger,
            surface_loader,

            device_memory_properties,
            present_queue,

            surface,
            surface_format,

            depth_format,

            swapchain,
            present_mode,
            present_image_views,

            cmd_pool,
        })
    }

    pub fn recreate_swapchain(&mut self, image_extent: vk::Extent2D) -> bool {
        let surface_capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap()
        };

        let recreated = self.swapchain.recreate(
            &self.device,
            self.surface,
            self.surface_format,
            &surface_capabilities,
            self.present_mode,
            image_extent,
        );

        if recreated {
            self.update_present_image_views();
        }

        recreated
    }

    fn update_present_image_views(&mut self) {
        for view in &self.present_image_views {
            unsafe {
                self.device.destroy_image_view(*view, None);
            }
        }
        self.present_image_views.clear();

        create_present_image_views(
            &self.device,
            self.swapchain.present_images(),
            self.surface_format,
            &mut self.present_image_views,
        )
    }

    pub fn destruct(&mut self) {
        unsafe {
            for view in &self.present_image_views {
                self.device.destroy_image_view(*view, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.swapchain.destruct();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_util_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.inst.destroy_instance(None);
        }
    }
}
