mod buffer;
mod common;
mod image;
mod mesh;
mod swapchain;
mod util;

use ::ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr,
    util::{Align, read_spv},
    vk,
};
use ::winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
use buffer::Buffer;
use common::Vertex;
use image::Image;
use mesh::Mesh;
use std::{borrow::Cow, error::Error, ffi, fs::File, io::Cursor, os::raw::c_char, u32, vec::Vec};
use swapchain::Swapchain;
use util::find_memorytype_index;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

fn array_size<T>(a: &[T]) -> usize {
    a.len() * std::mem::size_of::<T>()
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("?")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("?")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "[{message_severity:?}:{message_type:?}] [{message_id_name} ({message_id_number})] : {message}\n",
    );

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

struct VkBase {
    pub inst: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub debug_util_loader: debug_utils::Instance,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub surface_loader: khr::surface::Instance,

    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
    pub present_queue: vk::Queue,

    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,

    pub swapchain: Swapchain,
    present_mode: vk::PresentModeKHR,
    present_image_views: Vec<vk::ImageView>,

    pub cmd_pool: vk::CommandPool,
    pub cmd_buf_draw: vk::CommandBuffer,
    pub cmd_buf_setup: vk::CommandBuffer,

    depth_image: Image,

    pub fence_setup_cmd_reuse: vk::Fence,
    pub frame_fences: Vec<vk::Fence>,

    pub present_complete_semaphores: Vec<vk::Semaphore>,
    pub render_complete_semaphores: Vec<vk::Semaphore>,

    pub max_frames_inflight: usize,
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
                .api_version(vk::make_api_version(0, 1, 0, 1));

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

        let device = {
            let device_extension_names_raw = [khr::swapchain::NAME.as_ptr()];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);
            let device_createinfo = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features);
            unsafe {
                inst.create_device(physical_device, &device_createinfo, None)
                    .unwrap()
            }
        };

        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let surface_format = pick_present_image_format(&surface_formats);
        println!("[DEBUG LINW] picked surface format: {:?}", surface_format);

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

        let (cmd_buf_setup, cmd_buf_draw) = unsafe {
            let cmd_buf_allocateinfo = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(2)
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY);
            let cmd_bufs = device
                .allocate_command_buffers(&cmd_buf_allocateinfo)
                .unwrap();
            (cmd_bufs[0], cmd_bufs[1])
        };

        let device_memory_properties =
            unsafe { inst.get_physical_device_memory_properties(physical_device) };

        let depth_image = Image::default();
        // Image::new_depth_image(&device, device_memory_properties, swapchain.image_extent());

        let fence_setup_cmd_reuse = unsafe {
            let create_info = vk::FenceCreateInfo::default();
            device
                .create_fence(&create_info, None)
                .expect("Failed to create fence.")
        };

        let max_frames_inflight = 1; // present_image_views.len();

        let mut frame_fences = Vec::with_capacity(max_frames_inflight);
        for _ in 0..max_frames_inflight {
            let create_info = vk::FenceCreateInfo::default();
            let fence = unsafe {
                device
                    .create_fence(&create_info, None)
                    .expect("Failed to create fence.")
            };
            frame_fences.push(fence);
        }

        let mut present_complete_semaphores = Vec::with_capacity(max_frames_inflight);
        let mut render_complete_semaphores = Vec::with_capacity(max_frames_inflight);
        for _ in 0..max_frames_inflight {
            let createinfo = vk::SemaphoreCreateInfo::default();
            let (present_sema, render_sema) = unsafe {
                (
                    device.create_semaphore(&createinfo, None).unwrap(),
                    device.create_semaphore(&createinfo, None).unwrap(),
                )
            };
            present_complete_semaphores.push(present_sema);
            render_complete_semaphores.push(render_sema);
        }

        Ok(Self {
            inst,
            physical_device,
            device,
            debug_util_loader,
            debug_messenger,
            surface_loader,

            device_memory_properties,
            queue_family_index,
            present_queue,

            surface,
            surface_format,

            swapchain,
            present_mode,
            present_image_views,

            cmd_pool,
            cmd_buf_draw,
            cmd_buf_setup,

            depth_image,

            fence_setup_cmd_reuse,
            frame_fences,

            present_complete_semaphores,
            render_complete_semaphores,

            max_frames_inflight,
        })
    }

    pub fn record_submit_cmd_buf<F: FnOnce(&Device, vk::CommandBuffer)>(
        &self,
        cmd_buf: vk::CommandBuffer,
        frame_fence: vk::Fence,
        submit_queue: vk::Queue,
        wait_mask: &[vk::PipelineStageFlags],
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        record_cmd_buf_f: F,
    ) {
        unsafe {
            self.device
                .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .expect("Failed to reset command buffer.");

            let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
                .expect("Failed to begin command buffer recording.");

            record_cmd_buf_f(&self.device, cmd_buf);

            self.device
                .end_command_buffer(cmd_buf)
                .expect("Failed to end command buffer recording.");

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_mask)
                .command_buffers(std::slice::from_ref(&cmd_buf))
                .signal_semaphores(signal_semaphores);

            self.device
                .queue_submit(submit_queue, &[submit_info], frame_fence)
                .expect("Failed to queue submit.");
        }
    }

    pub fn setup(&self) {
        self.record_submit_cmd_buf(
            self.cmd_buf_setup,
            self.fence_setup_cmd_reuse,
            self.present_queue,
            &[],
            &[],
            &[],
            |device, cmd_buf_setup| {
                let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                    .image(self.depth_image.image)
                    .dst_access_mask(
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1)
                            .level_count(1),
                    );

                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd_buf_setup,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    );
                }
            },
        );
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
            // self.update_depth_image();
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

    pub fn update_depth_image(&mut self) {
        let device_memory_properties = unsafe {
            self.inst
                .get_physical_device_memory_properties(self.physical_device)
        };
        self.depth_image.destroy(&self.device);
        self.depth_image = Image::new_depth_image(
            &self.device,
            device_memory_properties,
            self.swapchain.image_extent(),
        );
    }
}

impl Drop for VkBase {
    fn drop(&mut self) {
        unsafe {
            for sema in &self.present_complete_semaphores {
                self.device.destroy_semaphore(*sema, None);
            }
            for sema in &self.render_complete_semaphores {
                self.device.destroy_semaphore(*sema, None);
            }
            for fence in &self.frame_fences {
                self.device.destroy_fence(*fence, None);
            }
            self.device.destroy_fence(self.fence_setup_cmd_reuse, None);
            self.depth_image.destroy(&self.device);
            for view in &self.present_image_views {
                self.device.destroy_image_view(*view, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.swapchain.destroy();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_util_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.inst.destroy_instance(None);
        }
    }
}

#[derive(Default)]
struct App {
    window_width: u32,
    window_height: u32,
    window: Option<Window>,
    vk_base: Option<VkBase>,

    renderpass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    mesh: Mesh,
    index_buffer: Buffer,
    vertex_buffer: Buffer,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
    viewports: Vec<vk::Viewport>,
    scissors: Vec<vk::Rect2D>,
    graphics_pipelines: Vec<vk::Pipeline>,

    frame_count: usize,
}

impl App {
    fn init_vk(&mut self) {
        self.vk_base = VkBase::new(self.window.as_ref().unwrap()).ok();
        // self.vk_base.as_ref().unwrap().setup();
    }

    fn destroy_vk(&mut self) {
        self.vk_base = None;
    }

    pub fn window_size(mut self, width: u32, height: u32) -> Self {
        self.window_width = width;
        self.window_height = height;
        self
    }

    pub fn prepare_pipeline(&mut self) {
        let vk_base = self.vk_base.as_ref().unwrap();

        self.renderpass = {
            let renderpass_attachments = [
                vk::AttachmentDescription {
                    format: vk_base.surface_format.format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    ..Default::default()
                },
                // vk::AttachmentDescription {
                //     format: vk::Format::D16_UNORM,
                //     samples: vk::SampleCountFlags::TYPE_1,
                //     load_op: vk::AttachmentLoadOp::CLEAR,
                //     initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                //     final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                //     ..Default::default()
                // },
            ];

            let color_attachment_refs = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let depth_attachment_refs = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };

            let dependencies = [vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags::NONE_KHR,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ..Default::default()
            }];

            let subpass = vk::SubpassDescription::default()
                .color_attachments(&color_attachment_refs)
                // .depth_stencil_attachment(&depth_attachment_refs)
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

            let renderpass_createinfo = vk::RenderPassCreateInfo::default()
                .attachments(&renderpass_attachments)
                .subpasses(std::slice::from_ref(&subpass))
                .dependencies(&dependencies);

            unsafe {
                vk_base
                    .device
                    .create_render_pass(&renderpass_createinfo, None)
                    .expect("Failed to create render pass.")
            }
        };

        let image_extent = vk_base.swapchain.image_extent();
        self.framebuffers = vk_base
            .present_image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments =
                    [present_image_view /*, vk_base.depth_image.view*/];
                let framebuffer_createinfo = vk::FramebufferCreateInfo::default()
                    .render_pass(self.renderpass)
                    .attachments(&framebuffer_attachments)
                    .width(image_extent.width)
                    .height(image_extent.height)
                    .layers(1);
                unsafe {
                    vk_base
                        .device
                        .create_framebuffer(&framebuffer_createinfo, None)
                        .expect("Failed to create frame buffer.")
                }
            })
            .collect();

        self.mesh = Mesh::from_obj("./assets/stanford-bunny.obj");

        self.index_buffer = Buffer::new(
            &vk_base.device,
            (size_of::<u32>() * self.mesh.indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &vk_base.device_memory_properties,
        );
        self.index_buffer.copy_data(&self.mesh.indices);

        self.vertex_buffer = Buffer::new(
            &vk_base.device,
            (size_of::<Vertex>() * self.mesh.vertices.len()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vk_base.device_memory_properties,
        );
        self.vertex_buffer.copy_data(&self.mesh.vertices);

        self.vertex_shader_module = unsafe {
            let vertex_spv = {
                let mut file =
                    Cursor::new(&include_bytes!("../shader/triangle/triangle.vert.spv")[..]);
                read_spv(&mut file).expect("Failed to read vertex shader spv file.")
            };
            let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&vertex_spv);
            vk_base
                .device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Failed to create vertex shader module")
        };
        self.fragment_shader_module = unsafe {
            let frag_spv = {
                let mut file =
                    Cursor::new(&include_bytes!("../shader/triangle/triangle.frag.spv")[..]);
                read_spv(&mut file).expect("Failed to read fragment shader spv file.")
            };
            let frag_shader_info = vk::ShaderModuleCreateInfo::default().code(&frag_spv);
            vk_base
                .device
                .create_shader_module(&frag_shader_info, None)
                .expect("Failed to create fragment shader module")
        };

        self.viewports = vec![vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: image_extent.width as f32,
            height: image_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        self.scissors = vec![image_extent.into()];

        self.graphics_pipelines = {
            self.pipeline_layout = unsafe {
                let layout_createinfo = vk::PipelineLayoutCreateInfo::default();
                vk_base
                    .device
                    .create_pipeline_layout(&layout_createinfo, None)
                    .unwrap()
            };

            let shader_entry_name = c"main";
            let shader_stage_createinfos = [
                vk::PipelineShaderStageCreateInfo {
                    module: self.vertex_shader_module,
                    p_name: shader_entry_name.as_ptr(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    module: self.fragment_shader_module,
                    p_name: shader_entry_name.as_ptr(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    ..Default::default()
                },
            ];

            let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
                binding: 0,
                stride: size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }];

            let vertex_input_attribute_descriptions = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Vertex, pos) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Vertex, color) as u32,
                },
            ];

            let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
                .vertex_binding_descriptions(&vertex_input_binding_descriptions);

            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
                .scissors(&self.scissors)
                .viewports(&self.viewports);

            let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                ..Default::default()
            };

            let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            let noop_stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                ..Default::default()
            };

            let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: 1,
                depth_write_enable: 1,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                front: noop_stencil_state,
                back: noop_stencil_state,
                max_depth_bounds: 1.0,
                ..Default::default()
            };

            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op(vk::LogicOp::CLEAR)
                .attachments(&color_blend_attachment_states);

            let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

            let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stage_createinfos)
                .vertex_input_state(&vertex_input_state_info)
                .input_assembly_state(&vertex_input_assembly_state_info)
                .viewport_state(&viewport_state_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_state_info)
                // .depth_stencil_state(&depth_state_info)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state_info)
                .layout(self.pipeline_layout)
                .render_pass(self.renderpass);

            unsafe {
                vk_base
                    .device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[graphic_pipeline_info],
                        None,
                    )
                    .expect("Unable to create graphics pipeline")
            }
        };
    }

    pub fn teardown_pipeline(&mut self) {
        let vk_base = self.vk_base.as_ref().unwrap();

        unsafe {
            vk_base.device.device_wait_idle().unwrap();
            for pipeline in &self.graphics_pipelines {
                vk_base.device.destroy_pipeline(*pipeline, None);
            }
            vk_base
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            vk_base
                .device
                .destroy_shader_module(self.vertex_shader_module, None);
            vk_base
                .device
                .destroy_shader_module(self.fragment_shader_module, None);
            self.index_buffer.destroy(&vk_base.device);
            self.vertex_buffer.destroy(&vk_base.device);
            for framebuffer in &self.framebuffers {
                vk_base.device.destroy_framebuffer(*framebuffer, None);
            }
            vk_base.device.destroy_render_pass(self.renderpass, None);
        }
    }

    pub fn update_framebuffers(&mut self) {
        let vk_base = self.vk_base.as_ref().unwrap();

        for framebuffer in &self.framebuffers {
            unsafe {
                vk_base.device.destroy_framebuffer(*framebuffer, None);
            }
        }
        self.framebuffers.clear();

        let image_extent = vk_base.swapchain.image_extent();
        for image_view in &vk_base.present_image_views {
            let framebuffer_attachments = [*image_view /*, vk_base.depth_image.view*/];
            let framebuffer_createinfo = vk::FramebufferCreateInfo::default()
                .render_pass(self.renderpass)
                .attachments(&framebuffer_attachments)
                .width(image_extent.width)
                .height(image_extent.height)
                .layers(1);
            let framebuffer = unsafe {
                vk_base
                    .device
                    .create_framebuffer(&framebuffer_createinfo, None)
                    .expect("Failed to create frame buffer.")
            };
            self.framebuffers.push(framebuffer);
        }
    }

    pub fn draw(&mut self) {
        let vk_base = if let Some(base) = self.vk_base.as_ref() {
            base
        } else {
            return;
        };

        let inflight_frame_index = self.frame_count % vk_base.max_frames_inflight;
        let present_complete_semaphore =
            vk_base.present_complete_semaphores[inflight_frame_index as usize];
        let render_complete_semaphore =
            vk_base.render_complete_semaphores[inflight_frame_index as usize];

        let present_index = vk_base
            .swapchain
            .acquire_next_image(present_complete_semaphore);

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [135.0 / 255.0, 206.0 / 255.0, 250.0 / 255.0, 15.0 / 255.0],
                },
            },
            // vk::ClearValue {
            //     depth_stencil: vk::ClearDepthStencilValue {
            //         depth: 1.0,
            //         stencil: 0,
            //     },
            // },
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.renderpass)
            .framebuffer(self.framebuffers[present_index as usize])
            .render_area(vk_base.swapchain.image_extent().into())
            .clear_values(&clear_values);

        let frame_fence = vk_base.frame_fences[inflight_frame_index];
        vk_base.record_submit_cmd_buf(
            vk_base.cmd_buf_draw,
            vk::Fence::null(),
            vk_base.present_queue,
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[present_complete_semaphore],
            &[render_complete_semaphore],
            |device, cmd_buf_draw| unsafe {
                device.cmd_begin_render_pass(
                    cmd_buf_draw,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                device.cmd_bind_pipeline(
                    cmd_buf_draw,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipelines[0],
                );

                let image_extent = vk_base.swapchain.image_extent();
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: image_extent.width as f32,
                    height: image_extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let scissor: vk::Rect2D = image_extent.into();
                device.cmd_set_viewport(cmd_buf_draw, 0, &[viewport]);
                device.cmd_set_scissor(cmd_buf_draw, 0, &[scissor]);

                device.cmd_bind_vertex_buffers(cmd_buf_draw, 0, &[self.vertex_buffer.buf], &[0]);

                device.cmd_bind_index_buffer(
                    cmd_buf_draw,
                    self.index_buffer.buf,
                    0,
                    vk::IndexType::UINT32,
                );

                device.cmd_draw_indexed(cmd_buf_draw, self.mesh.indices.len() as u32, 1, 0, 0, 1);

                device.cmd_end_render_pass(cmd_buf_draw);
            },
        );

        vk_base.swapchain.queue_present(
            vk_base.present_queue,
            render_complete_semaphore,
            present_index,
        );

        /*
        if frame_count >= vk_base.max_frames_inflight {
            let wait_index = (frame_count + 1) % vk_base.max_frames_inflight;
            let wait_fence = vk_base.frame_fences[wait_index];

            unsafe {
                vk_base
                    .device
                    .wait_for_fences(&[wait_fence], true, u64::MAX)
                    .unwrap();
                vk_base.device.reset_fences(&[wait_fence]).unwrap();
            }
        }
        */

        unsafe {
            vk_base.device.device_wait_idle().unwrap();
        }

        assert!(self.frame_count < usize::MAX);
        self.frame_count += 1;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(winit::dpi::PhysicalSize::new(
                            self.window_width,
                            self.window_height,
                        ))
                        .with_title("Acadia"),
                )
                .unwrap(),
        );

        self.init_vk();
        self.prepare_pipeline();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("[DEBUG LINW] close requested.");
                self.teardown_pipeline();
                self.destroy_vk();
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                println!("[DEBUG LINW] redraw requested.");
                self.draw();
            }
            WindowEvent::Resized(size) => {
                println!("[DEBUG LINW] resized requested");
                if let Some(vk_base) = self.vk_base.as_mut() {
                    let recreated = vk_base.recreate_swapchain(vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    });
                    if recreated {
                        self.update_framebuffers();
                    }
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default().window_size(1024, 768);

    let _result = event_loop.run_app(&mut app);
}
