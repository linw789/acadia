use crate::{
    image::ImagePool,
    shader::{Shader, load_shaders},
    vkbase::VkBase,
};
use ash::vk;
use std::{collections::HashMap, rc::Rc, sync::Arc};
use winit::window::Window;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    pub vkbase: VkBase,

    frame_fences: Vec<vk::Fence>,
    present_acquired_semaphores: Vec<vk::Semaphore>,
    render_complete_semaphores: Vec<vk::Semaphore>,

    pub cmd_bufs: Vec<vk::CommandBuffer>,

    pub desc_pool: vk::DescriptorPool,
    pub shader_set: HashMap<String, Rc<Shader>>,
    pub image_pool: ImagePool,

    pub depth_format: vk::Format,
    depth_image_index: u32,

    frame_count: u64,
    in_flight_frame_index: usize,
    present_image_index: usize,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let vkbase = VkBase::new(window).unwrap();

        assert!(MAX_FRAMES_IN_FLIGHT <= vkbase.swapchain.present_images().len());

        let cmd_bufs = unsafe {
            let allocinfo = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32)
                .command_pool(vkbase.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY);
            vkbase.device.allocate_command_buffers(&allocinfo).unwrap()
        };

        let frame_fences = unsafe {
            (0..MAX_FRAMES_IN_FLIGHT)
                .into_iter()
                .map(|_| {
                    let createinfo =
                        vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                    vkbase.device.create_fence(&createinfo, None).unwrap()
                })
                .collect()
        };

        let present_acquired_semaphores = unsafe {
            (0..MAX_FRAMES_IN_FLIGHT)
                .into_iter()
                .map(|_| {
                    let createinfo = vk::SemaphoreCreateInfo::default();
                    vkbase.device.create_semaphore(&createinfo, None).unwrap()
                })
                .collect()
        };

        let render_complete_semaphores = unsafe {
            (0..vkbase.swapchain.present_images().len())
                .into_iter()
                .map(|_| {
                    let createinfo = vk::SemaphoreCreateInfo::default();
                    vkbase.device.create_semaphore(&createinfo, None).unwrap()
                })
                .collect()
        };

        let desc_pool = {
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
                vkbase
                    .device
                    .create_descriptor_pool(&pool_createinfo, None)
                    .unwrap()
            }
        };

        let shader_set = load_shaders(&vkbase.device, "target/shaders/");

        let mut image_pool =
            ImagePool::new(Arc::clone(&vkbase.device), vkbase.device_memory_properties);

        let depth_format = vk::Format::D32_SFLOAT;
        let depth_image_index =
            image_pool.new_depth_image(vkbase.swapchain.image_extent(), depth_format);

        Self {
            vkbase,
            frame_fences,
            present_acquired_semaphores,
            render_complete_semaphores,

            cmd_bufs,

            desc_pool,
            shader_set,
            image_pool,

            depth_format,
            depth_image_index,

            frame_count: 0,
            in_flight_frame_index: 0,
            present_image_index: 0,
        }
    }

    pub fn depth_image_view(&self) -> vk::ImageView {
        self.image_pool.get_at_index(self.depth_image_index).view
    }

    pub fn present_image_view(&self) -> vk::ImageView {
        self.vkbase.present_image_views[self.present_image_index]
    }

    pub fn curr_cmd_cuf(&self) -> vk::CommandBuffer {
        self.cmd_bufs[self.in_flight_frame_index]
    }

    pub fn in_flight_frame_index(&self) -> usize {
        self.in_flight_frame_index
    }

    pub fn begin_frame(&mut self) {
        let present_acquired_semaphore =
            self.present_acquired_semaphores[self.in_flight_frame_index];
        let frame_fence = self.frame_fences[self.in_flight_frame_index];

        unsafe {
            self.vkbase
                .device
                .wait_for_fences(&[frame_fence], true, u64::MAX)
                .unwrap();
            self.vkbase.device.reset_fences(&[frame_fence]).unwrap();
        }

        // TODO: recreate swapchain if vkResult is OUT_OF_DATE.
        self.present_image_index = self
            .vkbase
            .swapchain
            .acquire_next_image(present_acquired_semaphore)
            .unwrap() as usize;

        let cmd_buf = self.cmd_bufs[self.in_flight_frame_index];
        let present_image = self.vkbase.swapchain.present_images()[self.present_image_index];

        // Re-start command buffer recording.
        unsafe {
            self.vkbase
                .device
                .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .unwrap();

            let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.vkbase
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
            self.vkbase
                .device
                .cmd_pipeline_barrier2(cmd_buf, &dependency_info);
        }

        // Transition the depth image to the layout DEPTH_ATTACHMENT_OPTIMAL.
        unsafe {
            let depth_image = self.image_pool.get_at_index(self.depth_image_index);
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
                .image(depth_image.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            self.vkbase
                .device
                .cmd_pipeline_barrier2(cmd_buf, &dependency_info);
        }
    }

    pub fn end_frame(&mut self) {
        let frame_fence = self.frame_fences[self.in_flight_frame_index];
        let present_acquired_semaphore =
            self.present_acquired_semaphores[self.in_flight_frame_index];
        let render_complete_semaphore = self.render_complete_semaphores[self.present_image_index];
        let present_image = self.vkbase.swapchain.present_images()[self.present_image_index];
        let cmd_buf = self.cmd_bufs[self.in_flight_frame_index];

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
            self.vkbase
                .device
                .cmd_pipeline_barrier2(cmd_buf, &dependency_info);
        }

        // End command buffer recording.
        unsafe {
            self.vkbase.device.end_command_buffer(cmd_buf).unwrap();
        }

        unsafe {
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(std::slice::from_ref(&present_acquired_semaphore))
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(std::slice::from_ref(&cmd_buf))
                .signal_semaphores(std::slice::from_ref(&render_complete_semaphore));

            self.vkbase
                .device
                .queue_submit(self.vkbase.present_queue, &[submit_info], frame_fence)
                .expect("Failed to queue submit.");
        }

        self.vkbase.swapchain.queue_present(
            self.vkbase.present_queue,
            render_complete_semaphore,
            self.present_image_index as u32,
        );

        assert!(self.frame_count < u64::MAX);
        self.frame_count += 1;

        // Because we need to use `in_flight_frame_index` before `begin_frame()` is called, we
        // update it at the end of a frame. The first usage of `in_flight_frame_index` will be the
        // initial value 0, which is fine.
        self.in_flight_frame_index = (self.frame_count % (MAX_FRAMES_IN_FLIGHT as u64)) as usize;
    }

    pub fn resize(&mut self, size: vk::Extent2D) {
        let recreated = self.vkbase.recreate_swapchain(size);
        if recreated {
            self.image_pool.delete_at_index(self.depth_image_index);
            self.depth_image_index = self
                .image_pool
                .new_depth_image(self.vkbase.swapchain.image_extent(), self.depth_format);
        }
    }

    pub fn destruct(&mut self) {
        unsafe {
            for (_name, shader) in self.shader_set.iter_mut() {
                Rc::get_mut(shader).unwrap().destruct(&self.vkbase.device);
            }

            for sema in self.present_acquired_semaphores.iter() {
                self.vkbase.device.destroy_semaphore(*sema, None);
            }
            for sema in self.render_complete_semaphores.iter() {
                self.vkbase.device.destroy_semaphore(*sema, None);
            }
            for fence in &self.frame_fences {
                self.vkbase.device.destroy_fence(*fence, None);
            }

            self.vkbase
                .device
                .destroy_descriptor_pool(self.desc_pool, None);
            self.desc_pool = vk::DescriptorPool::null();

            self.image_pool.destruct();

            self.vkbase.destruct();
        }
    }
}
