use crate::{
    buffer::Buffer,
    offset_of,
    pipeline::PipelineBuilder,
    renderer::{MAX_FRAMES_IN_FLIGHT, Renderer},
    scene::Scene,
    shader::Program,
};
use ::ash::{Device, vk};
use ::glam::{Mat3, Mat4, Vec3, vec3};
use std::{f32::consts::PI, rc::Rc};

#[repr(C, packed)]
struct CirclePoints {
    position: Vec3,
    color: Vec3,
}

#[derive(Default)]
pub struct GizmoRotate {
    program: Program,
    pipeline: vk::Pipeline,
    desc_sets: Vec<vk::DescriptorSet>,
    uniform_buf: Buffer,
    circle_segment_count: usize,
    circle_x: Vec<CirclePoints>,
    circle_y: Vec<CirclePoints>,
    circle_z: Vec<CirclePoints>,
    circles_buffer: Buffer,
}

impl GizmoRotate {
    const PER_FRAME_UNIFORM_DATA_SIZE: usize = 64;

    pub fn new(renderer: &Renderer) -> Self {
        let program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(
                    renderer
                        .shader_set
                        .get("gizmo/transform3d-rotate.vert")
                        .unwrap(),
                ),
                Rc::clone(
                    renderer
                        .shader_set
                        .get("gizmo/transform3d-rotate.frag")
                        .unwrap(),
                ),
            ],
        );

        let pipeline = {
            let vertex_binding_descs = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(size_of::<CirclePoints>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)];

            let vertex_attrib_descs = [
                // per-vertex attributes
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(CirclePoints, position) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(CirclePoints, color) as u32,
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
            .topology(vk::PrimitiveTopology::LINE_LIST)
            .line_width(3.0)
            .build()
        };

        let uniform_buf = Buffer::new(
            &renderer.vkbase.device,
            (Self::PER_FRAME_UNIFORM_DATA_SIZE * MAX_FRAMES_IN_FLIGHT) as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
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
                    .buffer(uniform_buf.buf)
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

        let circle_segment_count = 48;
        // We use line_list, each line has two ends.
        let max_circle_point_count = circle_segment_count * 2;

        let circle_x = Vec::<CirclePoints>::with_capacity(max_circle_point_count);
        let circle_y = Vec::<CirclePoints>::with_capacity(max_circle_point_count);
        let circle_z = Vec::<CirclePoints>::with_capacity(max_circle_point_count);

        let circles_buffer = Buffer::new(
            &renderer.vkbase.device,
            (size_of::<CirclePoints>() * max_circle_point_count * 3) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &renderer.vkbase.device_memory_properties,
        );

        Self {
            program,
            pipeline,
            desc_sets,
            uniform_buf,
            circle_segment_count,
            circle_x,
            circle_y,
            circle_z,
            circles_buffer,
        }
    }

    // https://siegelord.net/circle_draw
    fn gen_circles(&mut self, object_pos: Vec3, camera_pos: Vec3) {
        // Generate a circle of points around `axis_id` centered at (0, 0, 0).
        fn gen_points(
            points: &mut Vec<CirclePoints>,
            segment_count: usize,
            to_camera: &Vec3,
            tan_theta: f32,
            cos_theta: f32,
            axis_id: usize,
        ) {
            const PROJECTIONS: [Mat3; 3] = [
                Mat3::from_cols_array(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ];
            const PERP_TRANSFORMS: [Mat3; 3] = [
                Mat3::from_cols_array(&[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]),
                Mat3::from_cols_array(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
                Mat3::from_cols_array(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ];
            const TANGENT_TRANSFORMS: [Mat3; 3] = [
                Mat3::from_cols_array(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0]),
                Mat3::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                Mat3::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ];
            const FULL_CIRCLE_STARTS: [Vec3; 3] = [
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 0.0, 1.0),
                vec3(1.0, 0.0, 0.0),
            ];
            const COLORS: [Vec3; 3] = [
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 0.0, 1.0),
            ];

            points.clear();

            let color = COLORS[axis_id];

            // Project `to_camera` onto the plane of the circle to be drawn.
            let projected = PROJECTIONS[axis_id] * to_camera;
            let projected_len_sqr = projected.length_squared();
            let (full_circle, first_point) = if projected_len_sqr > 0.001 {
                // Radius is always 1.0.
                let normalized = projected / projected_len_sqr.sqrt();
                // Calculate the vector perpendicular to the normalized projected vector.
                // The end point of this perpendicular vector is the start of the visible half arch.
                let perp = PERP_TRANSFORMS[axis_id] * normalized;
                (false, perp)
            } else {
                (true, FULL_CIRCLE_STARTS[axis_id])
            };

            let mut point = first_point;
            let half_count = segment_count / 2;
            for i in 0..segment_count {
                let tangent = (TANGENT_TRANSFORMS[axis_id] * point) * tan_theta;
                let next_point = (point + tangent) * cos_theta;

                if (i < half_count) || (full_circle || ((i % 2 == 1) && (i != (segment_count - 1))))
                {
                    points.push(CirclePoints {
                        position: point,
                        color,
                    });
                    points.push(CirclePoints {
                        position: next_point,
                        color,
                    });
                }

                point = next_point;
            }
            // println!("axis_id: {}, point count: {}, full_circle: {}", axis_id, points.len(), full_circle);
        }

        let theta = 2.0 * PI / (self.circle_segment_count as f32);
        let tan_theta = theta.tan();
        let cos_theta = theta.cos();

        // Translate the camera into the object space, so that the circle center is at (0, 0, 0) for
        // easier calculation.
        let to_camera = camera_pos - object_pos;

        gen_points(
            &mut self.circle_x,
            self.circle_segment_count,
            &to_camera,
            tan_theta,
            cos_theta,
            0,
        );
        gen_points(
            &mut self.circle_y,
            self.circle_segment_count,
            &to_camera,
            tan_theta,
            cos_theta,
            1,
        );
        gen_points(
            &mut self.circle_z,
            self.circle_segment_count,
            &to_camera,
            tan_theta,
            cos_theta,
            2,
        );
    }

    pub fn update(
        &mut self,
        in_flight_frame_index: usize,
        pers_view_matrix: &Mat4,
        camera_pos: Vec3,
    ) {
        let per_frame_uniform_data_offset =
            in_flight_frame_index * Self::PER_FRAME_UNIFORM_DATA_SIZE;
        self.uniform_buf
            .copy_value(per_frame_uniform_data_offset, pers_view_matrix);

        self.gen_circles(vec3(0.0, 0.0, 0.0), camera_pos);

        let max_circle_point_count = self.circle_segment_count * 2;

        self.circles_buffer.copy_slice(0, &self.circle_x);
        self.circles_buffer.copy_slice(
            size_of::<CirclePoints>() * max_circle_point_count,
            &self.circle_y,
        );
        self.circles_buffer.copy_slice(
            size_of::<CirclePoints>() * max_circle_point_count * 2,
            &self.circle_z,
        );
    }

    pub fn draw(&self, renderer: &Renderer) {
        let color_attachment_infos = [vk::RenderingAttachmentInfo::default()
            .image_view(renderer.present_image_view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(renderer.depth_image_view())
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE);

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
                &[self.circles_buffer.buf],
                &[0],
            );

            renderer.vkbase.device.cmd_bind_descriptor_sets(
                cmd_buf,
                self.program.bind_point,
                self.program.pipeline_layout,
                0,
                &self.desc_sets,
                &[(renderer.in_flight_frame_index() * Self::PER_FRAME_UNIFORM_DATA_SIZE) as u32],
            );

            renderer
                .vkbase
                .device
                .cmd_draw(cmd_buf, self.circle_x.len() as u32, 1, 0, 0);
            renderer.vkbase.device.cmd_draw(
                cmd_buf,
                self.circle_y.len() as u32,
                1,
                (self.circle_segment_count * 2) as u32,
                0,
            );
            renderer.vkbase.device.cmd_draw(
                cmd_buf,
                self.circle_z.len() as u32,
                1,
                (self.circle_segment_count * 2 * 2) as u32,
                0,
            );

            renderer.vkbase.device.cmd_end_rendering(cmd_buf);
        }
    }

    pub fn destruct(&mut self, device: &Device) {
        self.circles_buffer.destruct(device);
        self.uniform_buf.destruct(device);
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
        self.program.destruct(device);
    }
}
