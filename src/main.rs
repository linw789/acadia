mod buffer;
mod camera;
mod common;
mod descriptors;
mod entity;
mod gui;
mod image;
mod mesh;
mod pipeline;
mod scene;
mod shader;
mod swapchain;
mod texture;
mod util;

use ::ash::{Device, Entry, Instance, ext::debug_utils, khr, vk};
use ::winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use buffer::Buffer;
use camera::{Camera, CameraBuilder};
use descriptors::Descriptors;
use glam::{Mat4, Vec3, Vec4, vec2};
use gui::{DevGui, Text};
use image::Image;
use pipeline::{create_default_graphics_pipeline, create_dev_gui_graphics_pipeline};
use scene::{Scene, SceneLoader};
use shader::{Program, Shader, load_shaders};
use std::{
    borrow::Cow, collections::HashMap, error::Error, f32::consts::PI, ffi, os::raw::c_char, rc::Rc,
    u32, vec::Vec,
};
use swapchain::Swapchain;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

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

struct VkBase {
    pub inst: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub debug_util_loader: debug_utils::Instance,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub surface_loader: khr::surface::Instance,

    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub present_queue: vk::Queue,

    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,

    depth_format: vk::Format,

    pub swapchain: Swapchain,
    present_mode: vk::PresentModeKHR,
    present_image_views: Vec<vk::ImageView>,

    pub cmd_pool: vk::CommandPool,

    depth_image: Image,
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

        let depth_image =
            Image::new_depth_image(&device, &device_memory_properties, swapchain.image_extent(), depth_format);

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

            depth_image,
        })
    }

    pub fn recreate_swapchain(&mut self, image_extent: vk::Extent2D) {
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
            self.update_depth_image();
        }
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
        self.depth_image.destruct(&self.device);
        self.depth_image = Image::new_depth_image(
            &self.device,
            &device_memory_properties,
            self.swapchain.image_extent(),
            self.depth_format,
        );
    }
}

impl Drop for VkBase {
    fn drop(&mut self) {
        unsafe {
            self.depth_image.destruct(&self.device);
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

const MAX_FRAMES_IN_FLIGHT: u64 = 2;

#[derive(Default)]
struct App {
    window_width: u32,
    window_height: u32,
    window: Option<Window>,
    vk_base: Option<VkBase>,

    desciptors: Descriptors,

    frame_data_buffer: Buffer,
    frame_fences: Vec<vk::Fence>,
    present_acquired_semaphores: Vec<vk::Semaphore>,
    render_complete_semaphores: Vec<vk::Semaphore>,
    draw_cmd_bufs: Vec<vk::CommandBuffer>,

    viewports: Vec<vk::Viewport>,
    scissors: Vec<vk::Rect2D>,

    shader_set: HashMap<String, Rc<Shader>>,
    default_program: Program,
    dev_gui_program: Program,

    default_pipeline: vk::Pipeline,
    dev_gui_pipeline: vk::Pipeline,

    dev_gui: DevGui,
    scene: Scene,

    frame_count: u64,

    pub camera: Camera,

    pub is_left_button_pressed: bool,
}

impl App {
    pub fn with_window_size(mut self, width: u32, height: u32) -> Self {
        self.window_width = width;
        self.window_height = height;
        self
    }

    fn init_vk(&mut self) {
        self.vk_base = VkBase::new(self.window.as_ref().unwrap()).ok();
    }

    fn destroy_vk(&mut self) {
        self.vk_base = None;
    }

    pub fn prepare_pipeline(&mut self) {
        let vk_base = self.vk_base.as_ref().unwrap();

        assert!(MAX_FRAMES_IN_FLIGHT <= (vk_base.swapchain.present_images().len() as u64));

        self.draw_cmd_bufs = unsafe {
            let allocinfo = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32)
                .command_pool(vk_base.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY);
            vk_base.device.allocate_command_buffers(&allocinfo).unwrap()
        };

        self.frame_fences = unsafe {
            (0..MAX_FRAMES_IN_FLIGHT)
                .into_iter()
                .map(|_| {
                    let createinfo =
                        vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                    vk_base.device.create_fence(&createinfo, None).unwrap()
                })
                .collect()
        };

        self.present_acquired_semaphores = unsafe {
            (0..MAX_FRAMES_IN_FLIGHT)
                .into_iter()
                .map(|_| {
                    let createinfo = vk::SemaphoreCreateInfo::default();
                    vk_base.device.create_semaphore(&createinfo, None).unwrap()
                })
                .collect()
        };

        self.render_complete_semaphores = unsafe {
            (0..vk_base.swapchain.present_images().len())
                .into_iter()
                .map(|_| {
                    let createinfo = vk::SemaphoreCreateInfo::default();
                    vk_base.device.create_semaphore(&createinfo, None).unwrap()
                })
                .collect()
        };

        let image_extent = vk_base.swapchain.image_extent();

        let max_sampler_anisotropy = unsafe {
            let properties = vk_base
                .inst
                .get_physical_device_properties(vk_base.physical_device);
            properties.limits.max_sampler_anisotropy
        };

        let screen_size = vec2(self.window_width as f32, self.window_height as f32);
        self.dev_gui = DevGui::new(screen_size);
        self.dev_gui.load_font_texture(
            &vk_base.device,
            &vk_base.device_memory_properties,
            max_sampler_anisotropy,
            self.draw_cmd_bufs[0],
            vk_base.present_queue,
        );

        self.shader_set = load_shaders(&vk_base.device, "target/shaders/");
        self.default_program = Program::new(
            &vk_base.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(self.shader_set.get("default.vert").unwrap()),
                Rc::clone(self.shader_set.get("default.frag").unwrap()),
            ],
        );
        self.dev_gui_program = Program::new(
            &vk_base.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(self.shader_set.get("devgui-text.vert").unwrap()),
                Rc::clone(self.shader_set.get("devgui-text.frag").unwrap()),
            ],
        );

        self.camera = CameraBuilder::new()
            .position(Vec3::new(0.0, 0.0, 1.0))
            .up(Vec3::new(0.0, 1.0, 0.0))
            .lookat(Vec3::new(0.0, 0.0, -1.0))
            .fov_y(40.0 / 180.0 * std::f32::consts::PI)
            .near_z(0.1)
            .build()
            .unwrap();

        // let camera_transform_size = size_of::<Mat4>();
        // let light_data_size = size_of::<Vec4>();
        let frame_data_size = 96; // camera_transform_size + light_data_size;
        let frame_buffer_size = frame_data_size * MAX_FRAMES_IN_FLIGHT;
        self.frame_data_buffer = Buffer::new(
            &vk_base.device,
            frame_buffer_size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &vk_base.device_memory_properties,
        );

        self.viewports = vec![vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: image_extent.width as f32,
            height: image_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        self.dev_gui.text(&Text {
            text: "Hello World!".to_owned(),
            start: vec2(30.0, 30.0),
            height: 100.0,
        });

        self.dev_gui
            .build_vertex_index_buffer(&vk_base.device, &vk_base.device_memory_properties);

        self.scissors = vec![image_extent.into()];

        self.scene = SceneLoader::new(
            &vk_base.device,
            &vk_base.device_memory_properties,
            max_sampler_anisotropy,
            self.draw_cmd_bufs[0],
            vk_base.present_queue,
        )
        .load_mario();

        self.desciptors = Descriptors::new(&vk_base.device, self.scene.max_submesh_count());

        self.desciptors.update_per_frame_set(
            &vk_base.device,
            &self.frame_data_buffer,
            frame_data_size,
        );

        self.desciptors
            .update_dev_gui_sampler_set(&vk_base.device, &self.dev_gui.textures[0]);

        for (i, texture) in self.scene.textures.iter().enumerate() {
            self.desciptors
                .update_sampler_set(&vk_base.device, i, texture);
        }

        let color_attachment_formats = [vk_base.surface_format.format];

        self.default_pipeline = create_default_graphics_pipeline(
            &vk_base.device,
            &color_attachment_formats,
            vk_base.depth_format,
            &self.default_program,
        );
        self.dev_gui_pipeline = create_dev_gui_graphics_pipeline(
            &vk_base.device,
            &color_attachment_formats,
            vk_base.depth_format,
            &self.dev_gui_program,
        );
    }

    pub fn teardown_pipeline(&mut self) {
        let vk_base = self.vk_base.as_ref().unwrap();

        unsafe {
            vk_base.device.device_wait_idle().unwrap();
            self.scene.destruct(&vk_base.device);
            vk_base.device.destroy_pipeline(self.default_pipeline, None);
            vk_base.device.destroy_pipeline(self.dev_gui_pipeline, None);
            self.frame_data_buffer.destruct(&vk_base.device);
            self.desciptors.destruct(&vk_base.device);
            self.dev_gui.destruct(&vk_base.device);
            self.default_program.destruct(&vk_base.device);
            self.dev_gui_program.destruct(&vk_base.device);
            for (_name, shader) in self.shader_set.iter_mut() {
                Rc::get_mut(shader).unwrap().destruct(&vk_base.device);
            }
            for sema in &self.present_acquired_semaphores {
                vk_base.device.destroy_semaphore(*sema, None);
            }
            for sema in &self.render_complete_semaphores {
                vk_base.device.destroy_semaphore(*sema, None);
            }
            for fence in &self.frame_fences {
                vk_base.device.destroy_fence(*fence, None);
            }
        }
    }

    pub fn update_uniform_buffer(&self, in_flight_frame_index: usize) {
        let vk_base = self.vk_base.as_ref().unwrap();

        let image_extent = vk_base.swapchain.image_extent();
        let view_matrix = self.camera.view_matrix();
        let pers_matrix = self
            .camera
            .perspective_matrix((image_extent.width as f32) / (image_extent.height as f32));
        // Compensate for Vulkan NDC's y-axis being pointing downwards.
        let negative_y_matrix = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        let vp_matrix = [negative_y_matrix * pers_matrix * view_matrix];

        let camera_transform_size = size_of::<Mat4>();
        // let light_data_size = size_of::<Vec4>();
        let frame_data_size = 96; // camera_transform_size + light_data_size;
        let frame_data_offset = in_flight_frame_index * frame_data_size;
        self.frame_data_buffer
            .copy_data(frame_data_offset, &vp_matrix);

        let light_data = [Vec4::from((self.camera.lookat_dir(), 1.0))];
        self.frame_data_buffer
            .copy_data(frame_data_offset + camera_transform_size, &light_data);
    }

    fn render_default_pipeline(
        &self,
        cmd_buf: vk::CommandBuffer,
        present_image_view: vk::ImageView,
        in_flight_frame_index: usize,
    ) {
        let vk_base = self.vk_base.as_ref().unwrap();

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(present_image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [135.0 / 255.0, 206.0 / 255.0, 250.0 / 255.0, 15.0 / 255.0],
                },
            });
        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(vk_base.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk_base.swapchain.image_extent(),
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info))
            .depth_attachment(&depth_attachment_info);

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

        unsafe {
            vk_base.device.cmd_begin_rendering(cmd_buf, &rendering_info);

            vk_base.device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.default_pipeline,
            );

            vk_base.device.cmd_set_viewport(cmd_buf, 0, &[viewport]);
            vk_base.device.cmd_set_scissor(cmd_buf, 0, &[scissor]);

            for entity in &self.scene.entities {
                let mesh = &self.scene.meshes[entity.mesh_index as usize];
                vk_base
                    .device
                    .cmd_bind_vertex_buffers(cmd_buf, 0, &[mesh.vertex_buffer.buf], &[0]);

                vk_base.device.cmd_bind_index_buffer(
                    cmd_buf,
                    mesh.index_buffer.buf,
                    0,
                    vk::IndexType::UINT32,
                );

                // let camera_transform_size = size_of::<Mat4>();
                // let light_data_size = size_of::<Vec4>();
                let frame_data_size = 96; // camera_transform_size + light_data_size;
                vk_base.device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    self.default_program.bind_point,
                    self.default_program.pipeline_layout,
                    0,
                    &[self.desciptors.per_frame_data_set()],
                    &[(in_flight_frame_index * frame_data_size) as u32],
                );

                for (submesh, texture_i) in mesh.submeshes.iter().zip(entity.texture_indices.iter())
                {
                    vk_base.device.cmd_bind_descriptor_sets(
                        cmd_buf,
                        self.default_program.bind_point,
                        self.default_program.pipeline_layout,
                        1,
                        &[self.desciptors.sampler_set(*texture_i as usize)],
                        &[],
                    );

                    vk_base.device.cmd_draw_indexed(
                        cmd_buf,
                        submesh.index_count,
                        1,
                        submesh.index_offset,
                        submesh.vertex_offset,
                        1,
                    );
                }
            }

            vk_base.device.cmd_end_rendering(cmd_buf);
        }
    }

    fn render_dev_gui_pipeline(
        &self,
        cmd_buf: vk::CommandBuffer,
        present_image_view: vk::ImageView,
    ) {
        let vk_base = self.vk_base.as_ref().unwrap();

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(present_image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk_base.swapchain.image_extent(),
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info));

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

        unsafe {
            vk_base.device.cmd_begin_rendering(cmd_buf, &rendering_info);

            vk_base.device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.dev_gui_pipeline,
            );

            vk_base.device.cmd_set_viewport(cmd_buf, 0, &[viewport]);
            vk_base.device.cmd_set_scissor(cmd_buf, 0, &[scissor]);

            vk_base.device.cmd_bind_vertex_buffers(
                cmd_buf,
                0,
                &[self.dev_gui.vertex_buffer.buf],
                &[0],
            );

            vk_base.device.cmd_bind_index_buffer(
                cmd_buf,
                self.dev_gui.index_buffer.buf,
                0,
                vk::IndexType::UINT32,
            );

            vk_base.device.cmd_bind_descriptor_sets(
                cmd_buf,
                self.dev_gui_program.bind_point,
                self.dev_gui_program.pipeline_layout,
                0,
                &[self.desciptors.dev_gui_sampler_set()],
                &[],
            );

            vk_base
                .device
                .cmd_draw_indexed(cmd_buf, self.dev_gui.indices.len() as u32, 1, 0, 0, 1);

            vk_base.device.cmd_end_rendering(cmd_buf);
        }
    }

    fn record_command_buffer(&self, present_image_index: usize, in_flight_frame_index: usize) {
        let vk_base = self.vk_base.as_ref().unwrap();

        let cmd_buf = self.draw_cmd_bufs[in_flight_frame_index];
        let present_image = vk_base.swapchain.present_images()[present_image_index];
        let present_image_view = vk_base.present_image_views[present_image_index];

        // Re-start command buffer recording.
        unsafe {
            vk_base
                .device
                .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .unwrap();

            let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            vk_base
                .device
                .begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
                .expect("Failed to begin command buffer recording.");
        }

        // First transition the present image to the layout COLOR_ATTACHMENT_OPTIMAL.
        unsafe {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(present_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            vk_base
                .device
                .cmd_pipeline_barrier2(cmd_buf, &dependency_info);
        }

        // Transition the depth image to the layout DEPTH_ATTACHMENT_OPTIMAL.
        unsafe {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_base.depth_image.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            vk_base
                .device
                .cmd_pipeline_barrier2(cmd_buf, &dependency_info);
        }

        self.render_default_pipeline(cmd_buf, present_image_view, in_flight_frame_index);
        self.render_dev_gui_pipeline(cmd_buf, present_image_view);

        // After rendering, transition the present image to the layout PRESENT_SRC_KHR.
        unsafe {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(present_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            vk_base
                .device
                .cmd_pipeline_barrier2(cmd_buf, &dependency_info);
        }

        // End command buffer recording.
        unsafe {
            vk_base.device.end_command_buffer(cmd_buf).unwrap();
        }
    }

    pub fn draw_frame(&mut self) {
        let vk_base = if let Some(base) = self.vk_base.as_ref() {
            base
        } else {
            return;
        };

        let in_flight_frame_index = (self.frame_count % MAX_FRAMES_IN_FLIGHT) as usize;
        let present_acquired_semaphore = self.present_acquired_semaphores[in_flight_frame_index];
        let frame_fence = self.frame_fences[in_flight_frame_index];

        unsafe {
            vk_base
                .device
                .wait_for_fences(&[frame_fence], true, u64::MAX)
                .unwrap();
            vk_base.device.reset_fences(&[frame_fence]).unwrap();
        }

        // TODO: recreate swapchain if vkResult is OUT_OF_DATE.
        let present_image_index = vk_base
            .swapchain
            .acquire_next_image(present_acquired_semaphore)
            .unwrap() as usize;

        let render_complete_semaphore = self.render_complete_semaphores[present_image_index];

        self.update_uniform_buffer(in_flight_frame_index);
        self.record_command_buffer(present_image_index, in_flight_frame_index);

        unsafe {
            let cmd_buf = self.draw_cmd_bufs[in_flight_frame_index];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(std::slice::from_ref(&present_acquired_semaphore))
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(std::slice::from_ref(&cmd_buf))
                .signal_semaphores(std::slice::from_ref(&render_complete_semaphore));

            vk_base
                .device
                .queue_submit(vk_base.present_queue, &[submit_info], frame_fence)
                .expect("Failed to queue submit.");
        }

        vk_base.swapchain.queue_present(
            vk_base.present_queue,
            render_complete_semaphore,
            present_image_index as u32,
        );

        assert!(self.frame_count < u64::MAX);
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
            WindowEvent::KeyboardInput { event, .. } => {
                let scale = 0.52;
                if event.state.is_pressed() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::ArrowLeft)
                        | PhysicalKey::Code(KeyCode::KeyA) => {
                            self.camera.translate_local(Vec3::new(-scale, 0.0, 0.0));
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight)
                        | PhysicalKey::Code(KeyCode::KeyD) => {
                            self.camera.translate_local(Vec3::new(scale, 0.0, 0.0));
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                            self.camera.translate_local(Vec3::new(0.0, 0.0, -scale));
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown)
                        | PhysicalKey::Code(KeyCode::KeyS) => {
                            self.camera.translate_local(Vec3::new(0.0, 0.0, scale));
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    match state {
                        ElementState::Pressed => self.is_left_button_pressed = true,
                        ElementState::Released => self.is_left_button_pressed = false,
                    }
                }
            }
            WindowEvent::CloseRequested => {
                println!("[DEBUG LINW] close requested.");
                self.teardown_pipeline();
                self.destroy_vk();
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.draw_frame();
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                println!(
                    "[DEBUG LINW] resized requested: (w: {}, h: {})",
                    size.width, size.height
                );
                if let Some(vk_base) = self.vk_base.as_mut() {
                    vk_base.recreate_swapchain(vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    });
                }
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.is_left_button_pressed {
                    let scale = 0.2;
                    let ry = scale * (delta.0 as f32) / 180.0 * PI;
                    let rx = scale * (delta.1 as f32) / 180.0 * PI;

                    self.camera.rotate_world_y(-ry);
                    self.camera.rotate_local_x(-rx);
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default().with_window_size(1920, 1080);

    let _result = event_loop.run_app(&mut app);
}
