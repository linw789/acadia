use crate::{
    buffer::Buffer,
    common::Vertex,
    mesh::Mesh,
    offset_of,
    pipeline::PipelineBuilder,
    renderer::{MAX_FRAMES_IN_FLIGHT, Renderer},
    shader::Program,
};
use ::ash::{Device, vk};
use ::glam::{Mat4, Vec3, vec3};
use std::{f32::consts::PI, rc::Rc, vec::Vec};

#[repr(C, packed)]
struct GizmoTranslateInstance {
    id: u32,
    color: Vec3,
    transform: Mat4,
}

#[derive(Default)]
pub struct GizmoTranslate {
    program: Program,
    arrow: Mesh,
    pipeline: vk::Pipeline,
    desc_sets: Vec<vk::DescriptorSet>,
    mesh_instance_buffer: Buffer,
    uniform_buffer: Buffer,
}

impl GizmoTranslate {
    const PER_FRAME_UNIFORM_DATA_SIZE: usize = 64;
    const INSTANCE_COUNT: usize = 3;

    pub fn new(renderer: &Renderer) -> Self {
        let program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(
                    renderer
                        .shader_set
                        .get("gizmo/transform3d-translate.vert")
                        .unwrap(),
                ),
                Rc::clone(
                    renderer
                        .shader_set
                        .get("gizmo/transform3d-translate.frag")
                        .unwrap(),
                ),
            ],
            0,
        );

        let arrow = Mesh::from_obj(
            &renderer.vkbase.device,
            &renderer.vkbase.device_memory_properties,
            "assets/meshes/gizmo-arrow.obj",
        );

        let pipeline = {
            let vertex_binding_descs = [
                vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(size_of::<Vertex>() as u32)
                    .input_rate(vk::VertexInputRate::VERTEX),
                vk::VertexInputBindingDescription::default()
                    .binding(1)
                    .stride(size_of::<GizmoTranslateInstance>() as u32)
                    .input_rate(vk::VertexInputRate::INSTANCE),
            ];

            let vertex_attrib_descs = [
                // per-vertex attributes
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(Vertex, pos) as u32,
                },
                // per-instance attributes
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 1,
                    format: vk::Format::R32_UINT,
                    offset: offset_of!(GizmoTranslateInstance, id) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 1,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GizmoTranslateInstance, color) as u32,
                },
                // The first column of the transform matrix.
                vk::VertexInputAttributeDescription {
                    location: 3,
                    binding: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(GizmoTranslateInstance, transform) as u32,
                },
                // The second column of the transform matrix.
                vk::VertexInputAttributeDescription {
                    location: 4,
                    binding: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: (offset_of!(GizmoTranslateInstance, transform) as u32) + 16,
                },
                // The third column of the transform matrix.
                vk::VertexInputAttributeDescription {
                    location: 5,
                    binding: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: (offset_of!(GizmoTranslateInstance, transform) as u32) + 32,
                },
                // The fourth column of the transform matrix.
                vk::VertexInputAttributeDescription {
                    location: 6,
                    binding: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: (offset_of!(GizmoTranslateInstance, transform) as u32) + 48,
                },
            ];

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_binding_descs)
                .vertex_attribute_descriptions(&vertex_attrib_descs);

            let blend_attachment_state = [
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA),
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA),
            ];

            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&blend_attachment_state);

            let obj_id_render_target_format = vk::Format::R32_UINT;

            PipelineBuilder::new(
                &renderer.vkbase.device,
                &program,
                &vertex_input_state,
                &[
                    renderer.vkbase.surface_format.format,
                    obj_id_render_target_format,
                ],
                &color_blend_state,
            )
            .depth_format(renderer.vkbase.depth_format)
            .build()
        };

        let uniform_buffer = {
            Buffer::new(
                &renderer.vkbase.device,
                (Self::PER_FRAME_UNIFORM_DATA_SIZE * MAX_FRAMES_IN_FLIGHT) as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                &renderer.vkbase.device_memory_properties,
            )
        };

        let mesh_instance_buffer = Buffer::new(
            &renderer.vkbase.device,
            (size_of::<GizmoTranslateInstance>() * Self::INSTANCE_COUNT) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &renderer.vkbase.device_memory_properties,
        );

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
                    .buffer(uniform_buffer.buf)
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
            arrow,
            pipeline,
            desc_sets,
            mesh_instance_buffer,
            uniform_buffer,
        }
    }

    pub fn update(&mut self, in_flight_frame_index: usize, pers_view_matrix: &Mat4) {
        let per_frame_uniform_data_offset =
            in_flight_frame_index * Self::PER_FRAME_UNIFORM_DATA_SIZE;
        self.uniform_buffer
            .copy_value(per_frame_uniform_data_offset, pers_view_matrix);

        let scale = 0.5;
        let position = vec3(0.0, 0.0, 0.0);
        let translation = Mat4::from_translation(position);
        let scale = Mat4::from_scale(vec3(scale, scale, scale));

        let x_arrow_id = 1;
        let x_arrow_color = vec3(1.0, 0.0, 0.0);
        let x_arrow_transform = translation * Mat4::from_rotation_z(-0.5 * PI) * scale;

        let y_arrow_id = 2;
        let y_arrow_color = vec3(0.0, 1.0, 0.0);
        let y_arrow_transform = translation * scale;

        let z_arrow_id = 3;
        let z_arrow_color = vec3(0.0, 0.0, 1.0);
        let z_arrow_transform = translation * Mat4::from_rotation_x(0.5 * PI) * scale;

        let mut linear_copy = self.mesh_instance_buffer.linear_copy(0);
        linear_copy.copy_value(&x_arrow_id);
        linear_copy.copy_value(&x_arrow_color);
        linear_copy.copy_value(&x_arrow_transform);

        linear_copy.copy_value(&y_arrow_id);
        linear_copy.copy_value(&y_arrow_color);
        linear_copy.copy_value(&y_arrow_transform);

        linear_copy.copy_value(&z_arrow_id);
        linear_copy.copy_value(&z_arrow_color);
        linear_copy.copy_value(&z_arrow_transform);
    }

    pub fn draw(&self, renderer: &Renderer) {
        let color_attachment_infos = [
            vk::RenderingAttachmentInfo::default()
                .image_view(renderer.present_image_view())
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE),
            vk::RenderingAttachmentInfo::default()
                .image_view(renderer.obj_id_image_view())
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE),
        ];
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
                &[self.arrow.vertex_buffer.buf],
                &[0],
            );
            renderer.vkbase.device.cmd_bind_vertex_buffers(
                cmd_buf,
                1,
                &[self.mesh_instance_buffer.buf],
                &[0],
            );
            renderer.vkbase.device.cmd_bind_index_buffer(
                cmd_buf,
                self.arrow.index_buffer.buf,
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

            for submesh in &self.arrow.submeshes {
                renderer.vkbase.device.cmd_draw_indexed(
                    cmd_buf,
                    submesh.index_count,
                    Self::INSTANCE_COUNT as u32,
                    submesh.index_offset,
                    submesh.vertex_offset,
                    0,
                );
            }

            renderer.vkbase.device.cmd_end_rendering(cmd_buf);
        }
    }

    pub fn picked(&self, id: u32) -> Option<Vec3> {
        match id {
            1 => Some(vec3(1.0, 0.0, 0.0)),
            2 => Some(vec3(0.0, 1.0, 0.0)),
            3 => Some(vec3(0.0, 0.0, 1.0)),
            _ => None
        }
    }

    pub fn destruct(&mut self, device: &Device) {
        self.mesh_instance_buffer.destruct(device);
        self.uniform_buffer.destruct(device);
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
        self.arrow.destruct(device);
        self.program.destruct(device);
    }
}
