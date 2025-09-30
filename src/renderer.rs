use crate::{
    image::Image,
    shader::{Shader, load_shaders},
    vkbase::VkBase,
};
use ash::vk;
use std::{collections::HashMap, rc::Rc};
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

    frame_count: u64,
}

pub struct FrameIndex {
    pub present_image_index: usize,
    pub in_flight_frame_index: usize,
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

        Self {
            vkbase,
            frame_fences,
            present_acquired_semaphores,
            render_complete_semaphores,

            cmd_bufs,

            desc_pool,
            shader_set,

            frame_count: 0,
        }
    }

    pub fn begin_frame(&self) -> FrameIndex {
        let in_flight_frame_index = (self.frame_count % (MAX_FRAMES_IN_FLIGHT as u64)) as usize;
        let present_acquired_semaphore = self.present_acquired_semaphores[in_flight_frame_index];
        let frame_fence = self.frame_fences[in_flight_frame_index];

        unsafe {
            self.vkbase
                .device
                .wait_for_fences(&[frame_fence], true, u64::MAX)
                .unwrap();
            self.vkbase.device.reset_fences(&[frame_fence]).unwrap();
        }

        // TODO: recreate swapchain if vkResult is OUT_OF_DATE.
        let present_image_index = self
            .vkbase
            .swapchain
            .acquire_next_image(present_acquired_semaphore)
            .unwrap() as usize;

        let cmd_buf = self.cmd_bufs[in_flight_frame_index];
        let present_image = self.vkbase.swapchain.present_images()[present_image_index];

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
                .image(self.vkbase.depth_image.image)
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

        FrameIndex {
            present_image_index,
            in_flight_frame_index,
        }
    }

    pub fn end_frame(&mut self, frame_index: &FrameIndex) {
        let present_image_index = frame_index.present_image_index;

        let frame_fence = self.frame_fences[frame_index.in_flight_frame_index];
        let present_acquired_semaphore =
            self.present_acquired_semaphores[frame_index.in_flight_frame_index];
        let render_complete_semaphore = self.render_complete_semaphores[present_image_index];
        let present_image = self.vkbase.swapchain.present_images()[present_image_index];
        let cmd_buf = self.cmd_bufs[frame_index.in_flight_frame_index];

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
            present_image_index as u32,
        );

        assert!(self.frame_count < u64::MAX);
        self.frame_count += 1;
    }

    pub fn destruct(&mut self) {
        unsafe {
            self.vkbase
                .device
                .destroy_descriptor_pool(self.desc_pool, None);
            self.desc_pool = vk::DescriptorPool::null();

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

            self.vkbase.destruct();
        }
    }
}
