use ::ash::{Device, vk};
use ::glam::{Mat4, Vec3, vec3};
use ::winit::{
    dpi::PhysicalSize,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use acadia::{
    app::App,
    buffer::Buffer,
    camera::{Camera, CameraBuilder},
    common::{Vertex, Vertex2D, size_of_var},
    light::DirectionalLight,
    mesh::Mesh,
    offset_of,
    pipeline::PipelineBuilder,
    renderer::{MAX_FRAMES_IN_FLIGHT, Renderer},
    scene::Scene,
    shader::Program,
};
use std::rc::Rc;

#[derive(Default)]
struct ShadowPass {
    program: Program,
    pipeline: vk::Pipeline,
    desc_sets: Vec<vk::DescriptorSet>,
    per_frame_uniform_buf: Buffer,
    shadow_depth_image_handle: u32,
}

#[derive(Default)]
struct LightPass {
    program: Program,
    pipeline: vk::Pipeline,
    desc_sets: Vec<vk::DescriptorSet>,
    uniform_buf: Buffer,
}

#[derive(Default)]
struct ShadowViewPass {
    program: Program,
    pipeline: vk::Pipeline,
    desc_sets: Vec<vk::DescriptorSet>,

    vertex_buf: Buffer,
    index_buf: Buffer,
    index_count: u32,
}

#[derive(Default)]
struct ShadowTest {
    renderer: Option<Renderer>,

    shadow_pass: ShadowPass,
    light_pass: LightPass,
    shadow_view_pass: ShadowViewPass,

    mesh: Mesh,
    light: DirectionalLight,

    shadow_depth_image_size: vk::Extent2D,
    shadow_depth_image_handle: u32,
    shadow_depth_sampler: vk::Sampler,
    shadow_depth_view: vk::ImageView,
}

impl ShadowPass {
    const PER_FRAME_UNIFORM_DATA_SIZE: usize = 64;

    fn new(renderer: &Renderer, shadow_depth_image_handle: u32) -> Self {
        let program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![Rc::clone(
                renderer
                    .shader_set
                    .get("shadow-test/shadow-pass.vert")
                    .unwrap(),
            )],
        );

        let pipeline = {
            let vertex_binding_descs = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)];
            let vertex_attrib_descs = [vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            }];
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_binding_descs)
                .vertex_attribute_descriptions(&vertex_attrib_descs);

            let color_blend_state =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&[]);

            PipelineBuilder::new(
                &renderer.vkbase.device,
                &program,
                &vertex_input_state,
                &[],
                &color_blend_state,
            )
            .depth_format(renderer.vkbase.depth_format)
            .enable_dynamic_depth_bias(true)
            .build()
        };

        let per_frame_uniform_buf = {
            let per_frame_uniform_buf_total_size =
                Self::PER_FRAME_UNIFORM_DATA_SIZE * MAX_FRAMES_IN_FLIGHT;
            Buffer::new(
                &renderer.vkbase.device,
                per_frame_uniform_buf_total_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                &renderer.vkbase.device_memory_properties,
            )
        };

        let desc_sets = {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(program.desc_set_layouts())
                .descriptor_pool(renderer.desc_pool);

            unsafe {
                let desc_sets = renderer
                    .vkbase
                    .device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap();

                let desc_buf_infos = [vk::DescriptorBufferInfo::default()
                    .buffer(per_frame_uniform_buf.buf)
                    .offset(0)
                    .range(Self::PER_FRAME_UNIFORM_DATA_SIZE as u64)];
                let desc_writes = [vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .dst_set(desc_sets[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .buffer_info(&desc_buf_infos)];
                renderer
                    .vkbase
                    .device
                    .update_descriptor_sets(&desc_writes, &[]);

                desc_sets
            }
        };

        Self {
            program,
            pipeline,
            desc_sets,
            per_frame_uniform_buf,
            shadow_depth_image_handle,
        }
    }

    fn update_light_projection(&mut self, in_flight_frame_index: usize, light_proj: &Mat4) {
        self.per_frame_uniform_buf.copy_value(
            in_flight_frame_index * Self::PER_FRAME_UNIFORM_DATA_SIZE,
            light_proj,
        );
    }

    fn draw(&mut self, renderer: &Renderer, mesh: &Mesh, shadow_depth_image_size: vk::Extent2D) {
        let shadow_depth_image = renderer
            .image_pool
            .get_at_index(self.shadow_depth_image_handle);

        let depth_image_layout_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(shadow_depth_image.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        let pre_rendering_dependencies =
            vk::DependencyInfo::default().image_memory_barriers(&depth_image_layout_barriers);

        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(shadow_depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: shadow_depth_image_size,
            })
            .layer_count(1)
            .depth_attachment(&depth_attachment_info);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(shadow_depth_image_size.width as f32)
            .height(shadow_depth_image_size.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor: vk::Rect2D = shadow_depth_image_size.into();

        unsafe {
            let cmd_buf = renderer.curr_cmd_cuf();

            renderer
                .vkbase
                .device
                .cmd_pipeline_barrier2(cmd_buf, &pre_rendering_dependencies);

            renderer
                .vkbase
                .device
                .cmd_begin_rendering(cmd_buf, &rendering_info);

            renderer.vkbase.device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            renderer
                .vkbase
                .device
                .cmd_set_viewport(cmd_buf, 0, &[viewport]);
            renderer
                .vkbase
                .device
                .cmd_set_scissor(cmd_buf, 0, &[scissor]);

            renderer
                .vkbase
                .device
                .cmd_set_depth_bias(cmd_buf, 1.25, 0.0, -1.75);

            renderer.vkbase.device.cmd_bind_vertex_buffers(
                cmd_buf,
                0,
                &[mesh.vertex_buffer.buf],
                &[0],
            );
            renderer.vkbase.device.cmd_bind_index_buffer(
                cmd_buf,
                mesh.index_buffer.buf,
                0,
                vk::IndexType::UINT32,
            );

            renderer.vkbase.device.cmd_bind_descriptor_sets(
                cmd_buf,
                self.program.bind_point,
                self.program.pipeline_layout,
                0,
                &self.desc_sets,
                &[(renderer.in_flight_frame_index() * Self::PER_FRAME_UNIFORM_DATA_SIZE) as u32],
            );

            for submesh in &mesh.submeshes {
                renderer.vkbase.device.cmd_draw_indexed(
                    cmd_buf,
                    submesh.index_count,
                    1,
                    submesh.index_offset,
                    submesh.vertex_offset,
                    1,
                );
            }

            renderer.vkbase.device.cmd_end_rendering(cmd_buf);
        }
    }

    fn destruct(&mut self, device: &Device) {
        self.per_frame_uniform_buf.destruct(device);
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
        self.program.destruct(device);
    }
}

impl LightPass {
    const PER_FRAME_UNIFORM_DATA_SIZE: usize = 144;

    fn new(
        renderer: &Renderer,
        shadow_depth_sampler: vk::Sampler,
        shadow_depth_view: vk::ImageView,
    ) -> Self {
        let program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(
                    renderer
                        .shader_set
                        .get("shadow-test/light-pass.vert")
                        .unwrap(),
                ),
                Rc::clone(
                    renderer
                        .shader_set
                        .get("shadow-test/light-pass.frag")
                        .unwrap(),
                ),
            ],
        );

        let pipeline = {
            let vertex_binding_descs = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)];

            let vertex_attrib_descs = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(Vertex, pos) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Vertex, color) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(Vertex, normal) as u32,
                },
            ];

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_binding_descs)
                .vertex_attribute_descriptions(&vertex_attrib_descs);

            let color_attachment_state = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)];

            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&color_attachment_state);

            PipelineBuilder::new(
                &renderer.vkbase.device,
                &program,
                &vertex_input_state,
                &[renderer.vkbase.surface_format.format],
                &color_blend_state,
            )
            .depth_format(renderer.vkbase.depth_format)
            .build()
        };

        let uniform_buf = {
            Buffer::new(
                &renderer.vkbase.device,
                (Self::PER_FRAME_UNIFORM_DATA_SIZE * MAX_FRAMES_IN_FLIGHT) as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                &renderer.vkbase.device_memory_properties,
            )
        };

        let desc_sets = {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(program.desc_set_layouts())
                .descriptor_pool(renderer.desc_pool);

            unsafe {
                let desc_sets = renderer
                    .vkbase
                    .device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap();

                let desc_buf_infos = [vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buf.buf)
                    .offset(0)
                    .range(Self::PER_FRAME_UNIFORM_DATA_SIZE as u64)];

                let desc_image_infos = [vk::DescriptorImageInfo::default()
                    .sampler(shadow_depth_sampler)
                    .image_view(shadow_depth_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

                let desc_writes = [
                    vk::WriteDescriptorSet::default()
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .dst_set(desc_sets[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .buffer_info(&desc_buf_infos),
                    vk::WriteDescriptorSet::default()
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_set(desc_sets[0])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .image_info(&desc_image_infos),
                ];
                renderer
                    .vkbase
                    .device
                    .update_descriptor_sets(&desc_writes, &[]);

                desc_sets
            }
        };

        Self {
            program,
            pipeline,
            desc_sets,
            uniform_buf,
        }
    }

    fn update_uniform_buf(
        &mut self,
        in_flight_frame_index: usize,
        camera_proj: &Mat4,
        light_proj: &Mat4,
        light_dir: &Vec3,
    ) {
        let start_offset = (in_flight_frame_index * Self::PER_FRAME_UNIFORM_DATA_SIZE) as u64;
        let mut copy = self.uniform_buf.linear_copy(start_offset);
        copy.copy_value(camera_proj);
        copy.copy_value(light_proj);
        copy.copy_value(light_dir);
    }

    fn draw(&mut self, renderer: &Renderer, mesh: &Mesh, shadow_depth_image_handle: u32) {
        let shadow_depth_image = renderer.image_pool.get_at_index(shadow_depth_image_handle);
        let depth_image_layout_barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS_KHR,
            )
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            // NOTE, must set the old_layout to match the new_layout from the previous transition.
            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(shadow_depth_image.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        let pre_rendering_dependencies =
            vk::DependencyInfo::default().image_memory_barriers(&depth_image_layout_barrier);

        let color_attachment_infos = [vk::RenderingAttachmentInfo::default()
            .image_view(renderer.present_image_view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [135.0 / 255.0, 206.0 / 255.0, 250.0 / 255.0, 15.0 / 255.0],
                },
            })];
        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(renderer.depth_image_view())
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });

        let image_extent = renderer.vkbase.swapchain.image_extent();

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: image_extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachment_infos)
            .depth_attachment(&depth_attachment_info);

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
            let cmd_buf = renderer.curr_cmd_cuf();

            renderer
                .vkbase
                .device
                .cmd_pipeline_barrier2(cmd_buf, &pre_rendering_dependencies);

            renderer
                .vkbase
                .device
                .cmd_begin_rendering(cmd_buf, &rendering_info);

            renderer.vkbase.device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            renderer
                .vkbase
                .device
                .cmd_set_viewport(cmd_buf, 0, &[viewport]);
            renderer
                .vkbase
                .device
                .cmd_set_scissor(cmd_buf, 0, &[scissor]);

            renderer.vkbase.device.cmd_bind_vertex_buffers(
                cmd_buf,
                0,
                &[mesh.vertex_buffer.buf],
                &[0],
            );
            renderer.vkbase.device.cmd_bind_index_buffer(
                cmd_buf,
                mesh.index_buffer.buf,
                0,
                vk::IndexType::UINT32,
            );

            renderer.vkbase.device.cmd_bind_descriptor_sets(
                cmd_buf,
                self.program.bind_point,
                self.program.pipeline_layout,
                0,
                &self.desc_sets,
                &[(renderer.in_flight_frame_index() * Self::PER_FRAME_UNIFORM_DATA_SIZE) as u32],
            );

            for submesh in &mesh.submeshes {
                renderer.vkbase.device.cmd_draw_indexed(
                    cmd_buf,
                    submesh.index_count,
                    1,
                    submesh.index_offset,
                    submesh.vertex_offset,
                    1,
                );
            }

            renderer.vkbase.device.cmd_end_rendering(cmd_buf);
        }
    }

    fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
        self.uniform_buf.destruct(device);
        self.program.destruct(device);
    }
}

impl ShadowViewPass {
    fn new(
        renderer: &Renderer,
        shadow_depth_sampler: vk::Sampler,
        shadow_depth_view: vk::ImageView,
    ) -> Self {
        let program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(
                    renderer
                        .shader_set
                        .get("shadow-test/shadow-view.vert")
                        .unwrap(),
                ),
                Rc::clone(
                    renderer
                        .shader_set
                        .get("shadow-test/shadow-view.frag")
                        .unwrap(),
                ),
            ],
        );

        let pipeline = {
            let vertex_binding_descs = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(size_of::<Vertex2D>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)];

            let vertex_attrib_descs = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(Vertex2D, pos) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(Vertex2D, uv) as u32,
                },
            ];
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_binding_descs)
                .vertex_attribute_descriptions(&vertex_attrib_descs);

            let color_attachment_state = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&color_attachment_state);

            PipelineBuilder::new(
                &renderer.vkbase.device,
                &program,
                &vertex_input_state,
                &[renderer.vkbase.surface_format.format],
                &color_blend_state,
            )
            .build()
        };

        let desc_sets = if program.desc_set_layouts().len() > 0 {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(program.desc_set_layouts())
                .descriptor_pool(renderer.desc_pool);

            unsafe {
                let desc_sets = renderer
                    .vkbase
                    .device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap();

                let desc_image_infos = [vk::DescriptorImageInfo::default()
                    .sampler(shadow_depth_sampler)
                    .image_view(shadow_depth_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

                let desc_writes = [vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(desc_sets[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .image_info(&desc_image_infos)];
                renderer
                    .vkbase
                    .device
                    .update_descriptor_sets(&desc_writes, &[]);

                desc_sets
            }
        } else {
            Vec::new()
        };

        let vertices = [
            // top-left
            Vertex2D {
                pos: [-1.0, -1.0],
                color: [0.0, 0.0, 0.0, 0.0], // ignore color
                uv: [0.0, 0.0],
            },
            // bottom-left
            Vertex2D {
                pos: [-1.0, -0.25],
                color: [0.0, 0.0, 0.0, 0.0], // ignore color
                uv: [0.0, 1.0],
            },
            // bottom-right
            Vertex2D {
                pos: [-0.25, -0.25],
                color: [0.0, 0.0, 0.0, 0.0], // ignore color
                uv: [1.0, 1.0],
            },
            // top-right
            Vertex2D {
                pos: [-0.25, -1.0],
                color: [0.0, 0.0, 0.0, 0.0], // ignore color
                uv: [1.0, 0.0],
            },
        ];

        let indices = [0, 1, 3, 1, 2, 3];

        let vertex_buf = Buffer::from_slice(
            &renderer.vkbase.device,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &renderer.vkbase.device_memory_properties,
            &vertices,
        );

        let index_buf = Buffer::from_slice(
            &renderer.vkbase.device,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &renderer.vkbase.device_memory_properties,
            &indices,
        );
        let index_count = indices.len() as u32;

        Self {
            program,
            pipeline,
            desc_sets,
            vertex_buf,
            index_buf,
            index_count,
        }
    }

    fn draw(&self, renderer: &Renderer) {
        let color_attachment_infos = [vk::RenderingAttachmentInfo::default()
            .image_view(renderer.present_image_view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];

        let image_extent = renderer.vkbase.swapchain.image_extent();

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: image_extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachment_infos);

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
            let cmd_buf = renderer.curr_cmd_cuf();

            renderer
                .vkbase
                .device
                .cmd_begin_rendering(cmd_buf, &rendering_info);

            renderer.vkbase.device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            renderer
                .vkbase
                .device
                .cmd_set_viewport(cmd_buf, 0, &[viewport]);
            renderer
                .vkbase
                .device
                .cmd_set_scissor(cmd_buf, 0, &[scissor]);

            renderer.vkbase.device.cmd_bind_vertex_buffers(
                cmd_buf,
                0,
                &[self.vertex_buf.buf],
                &[0],
            );
            renderer.vkbase.device.cmd_bind_index_buffer(
                cmd_buf,
                self.index_buf.buf,
                0,
                vk::IndexType::UINT32,
            );

            if self.desc_sets.len() > 0 {
                renderer.vkbase.device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    self.program.bind_point,
                    self.program.pipeline_layout,
                    0,
                    &self.desc_sets,
                    &[],
                );
            }

            renderer
                .vkbase
                .device
                .cmd_draw_indexed(cmd_buf, self.index_count, 1, 0, 0, 0);

            renderer.vkbase.device.cmd_end_rendering(cmd_buf);
        }
    }

    fn destruct(&mut self, device: &Device) {
        self.index_buf.destruct(device);
        self.vertex_buf.destruct(device);
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
        self.program.destruct(device);
    }
}

impl Scene for ShadowTest {
    fn init(&mut self, window: &Window) {
        let mut renderer = Renderer::new(window);

        self.mesh = Mesh::from_obj(
            &renderer.vkbase.device,
            &renderer.vkbase.device_memory_properties,
            "assets/meshes/shadow-test.obj",
        );

        self.light = DirectionalLight::new(
            vec3(25.0, 25.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
        );

        self.shadow_depth_image_size = vk::Extent2D {
            width: 2048,
            height: 2048,
        };
        self.shadow_depth_image_handle = renderer
            .image_pool
            .new_depth_image(self.shadow_depth_image_size, renderer.vkbase.depth_format);

        self.shadow_depth_sampler = {
            let createinfo = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .mip_lod_bias(0.0)
                .anisotropy_enable(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS);

            unsafe {
                renderer
                    .vkbase
                    .device
                    .create_sampler(&createinfo, None)
                    .unwrap()
            }
        };

        self.shadow_depth_view = {
            let shadow_depth_image = renderer
                .image_pool
                .get_at_index(self.shadow_depth_image_handle);
            let createinfo = vk::ImageViewCreateInfo::default()
                .image(shadow_depth_image.image)
                .format(renderer.vkbase.depth_format)
                .view_type(vk::ImageViewType::TYPE_2D)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                );
            unsafe {
                renderer
                    .vkbase
                    .device
                    .create_image_view(&createinfo, None)
                    .unwrap()
            }
        };

        self.shadow_pass = ShadowPass::new(&renderer, self.shadow_depth_image_handle);
        self.light_pass =
            LightPass::new(&renderer, self.shadow_depth_sampler, self.shadow_depth_view);
        self.shadow_view_pass =
            ShadowViewPass::new(&renderer, self.shadow_depth_sampler, self.shadow_depth_view);

        self.renderer = Some(renderer);
    }

    fn update(&mut self, camera: &Camera) {
        let renderer = self.renderer.as_mut().unwrap();

        let image_extent = renderer.vkbase.swapchain.image_extent();
        let present_image_aspect_ratio = image_extent.width as f32 / image_extent.height as f32;
        let camera_proj = camera.ny_pers_view_matrix(present_image_aspect_ratio);

        let light_proj = self.light.ny_orthographic_projection(&self.mesh.bounds);

        self.shadow_pass
            .update_light_projection(renderer.in_flight_frame_index(), &light_proj);

        let light_direction = self.light.direction();
        self.light_pass.update_uniform_buf(
            renderer.in_flight_frame_index(),
            &camera_proj,
            &light_proj,
            &light_direction,
        );

        renderer.begin_frame();

        self.shadow_pass
            .draw(renderer, &self.mesh, self.shadow_depth_image_size);

        self.light_pass
            .draw(renderer, &self.mesh, self.shadow_depth_image_handle);

        self.shadow_view_pass.draw(renderer);

        renderer.end_frame();
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.resize(vk::Extent2D {
                width: size.width,
                height: size.height,
            });
        }
    }

    fn destruct(&mut self) {
        let device = &self.renderer.as_ref().unwrap().vkbase.device;
        unsafe {
            device.device_wait_idle().unwrap();

            device.destroy_image_view(self.shadow_depth_view, None);
            device.destroy_sampler(self.shadow_depth_sampler, None);
        }

        self.mesh.destruct(device);
        self.shadow_view_pass.destruct(device);
        self.light_pass.destruct(device);
        self.shadow_pass.destruct(device);

        self.renderer.as_mut().unwrap().destruct();
    }
}

fn main() {
    let camera = CameraBuilder::new()
        .position(vec3(-15.0, 5.0, 15.0))
        .up(vec3(0.0, 1.0, 0.0))
        .lookat(vec3(0.0, 0.0, 0.0))
        .fov_y(40.0 / 180.0 * std::f32::consts::PI)
        .near_z(0.1)
        .build()
        .unwrap();
    let mut app = App::new(
        PhysicalSize::<u32> {
            width: 1920,
            height: 1080,
        },
        Box::new(ShadowTest::default()),
        camera,
    );

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let _result = event_loop.run_app(&mut app);
}
