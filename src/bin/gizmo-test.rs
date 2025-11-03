use ::ash::vk;
use ::winit::{
    dpi::PhysicalSize,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use acadia::{
    app::App,
    buffer::Buffer,
    camera::{Camera, CameraBuilder},
    common::{Vertex, size_of_var},
    mesh::Mesh,
    offset_of,
    pipeline::PipelineBuilder,
    renderer::{MAX_FRAMES_IN_FLIGHT, Renderer},
    scene::Scene,
    shader::Program,
};
use glam::{Mat4, Vec3, vec3};
use std::{f32::consts::PI, rc::Rc};

#[repr(C, packed)]
struct GizmoAxisInstance {
    id: u32,
    color: Vec3,
    transform: Mat4,
}

#[derive(Default)]
struct GizmoTest {
    renderer: Option<Renderer>,
    program: Program,
    triangle: Mesh,
    arrow: Mesh,

    pipeline: vk::Pipeline,
    desc_set: vk::DescriptorSet,

    per_frame_uniform_buf: Buffer,
    instance_data_buffer: Buffer,
}

const PER_FRAME_UNIFORM_DATA_SIZE: usize = 64;
const INSTANCE_COUNT: usize = 3;

impl GizmoTest {
    fn draw_frame(&mut self) {
        let renderer = self.renderer.as_mut().unwrap();

        renderer.begin_frame();

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
                &[self.instance_data_buffer.buf],
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
                &[self.desc_set],
                &[(renderer.in_flight_frame_index() * PER_FRAME_UNIFORM_DATA_SIZE) as u32],
            );

            for submesh in &self.arrow.submeshes {
                renderer.vkbase.device.cmd_draw_indexed(
                    cmd_buf,
                    submesh.index_count,
                    INSTANCE_COUNT as u32,
                    submesh.index_offset,
                    submesh.vertex_offset,
                    0,
                );
            }

            renderer.vkbase.device.cmd_end_rendering(cmd_buf);
        }

        renderer.end_frame();
    }

    fn update_instance_data_buffer(&self, position: Vec3, scale: f32) {
        let translation = Mat4::from_translation(position);
        let scale = Mat4::from_scale(vec3(scale, scale, scale));

        let x_arrow_id = 0;
        let x_arrow_color = vec3(1.0, 0.0, 0.0);
        let x_arrow_transform = translation * Mat4::from_rotation_z(-0.5 * PI) * scale;

        let y_arrow_id = 1;
        let y_arrow_color = vec3(0.0, 1.0, 0.0);
        let y_arrow_transform = translation * scale;

        let z_arrow_id = 2;
        let z_arrow_color = vec3(0.0, 0.0, 1.0);
        let z_arrow_transform = translation * Mat4::from_rotation_x(0.5 * PI) * scale;

        let mut linear_copy = self.instance_data_buffer.linear_copy(0);
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
}

impl Scene for GizmoTest {
    fn init(&mut self, window: &Window) {
        let renderer = Renderer::new(window);
        self.program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(renderer.shader_set.get("gizmo.vert").unwrap()),
                Rc::clone(renderer.shader_set.get("gizmo.frag").unwrap()),
            ],
        );
        self.arrow = Mesh::from_obj(
            &renderer.vkbase.device,
            &renderer.vkbase.device_memory_properties,
            "assets/meshes/gizmo-arrow.obj",
        );

        self.pipeline = {
            let vertex_binding_descs = [
                vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(size_of::<Vertex>() as u32)
                    .input_rate(vk::VertexInputRate::VERTEX),
                vk::VertexInputBindingDescription::default()
                    .binding(1)
                    .stride(size_of::<GizmoAxisInstance>() as u32)
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
                    offset: offset_of!(GizmoAxisInstance, id) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 1,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GizmoAxisInstance, color) as u32,
                },
                // The first column of the transform matrix.
                vk::VertexInputAttributeDescription {
                    location: 3,
                    binding: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(GizmoAxisInstance, transform) as u32,
                },
                // The second column of the transform matrix.
                vk::VertexInputAttributeDescription {
                    location: 4,
                    binding: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: (offset_of!(GizmoAxisInstance, transform) as u32) + 16,
                },
                // The third column of the transform matrix.
                vk::VertexInputAttributeDescription {
                    location: 5,
                    binding: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: (offset_of!(GizmoAxisInstance, transform) as u32) + 32,
                },
                // The fourth column of the transform matrix.
                vk::VertexInputAttributeDescription {
                    location: 6,
                    binding: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: (offset_of!(GizmoAxisInstance, transform) as u32) + 48,
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
                &self.program,
                &vertex_input_state,
                &[renderer.vkbase.surface_format.format],
                &color_blend_state,
            )
            .depth_format(renderer.vkbase.depth_format)
            .build()
        };

        self.per_frame_uniform_buf = {
            Buffer::new(
                &renderer.vkbase.device,
                (PER_FRAME_UNIFORM_DATA_SIZE * MAX_FRAMES_IN_FLIGHT) as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                &renderer.vkbase.device_memory_properties,
            )
        };

        self.instance_data_buffer = Buffer::new(
            &renderer.vkbase.device,
            (size_of::<GizmoAxisInstance>() * INSTANCE_COUNT) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &renderer.vkbase.device_memory_properties,
        );

        self.desc_set = {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(self.program.desc_set_layouts())
                .descriptor_pool(renderer.desc_pool);

            unsafe {
                let desc_set = renderer
                    .vkbase
                    .device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap()[0];

                let desc_buf_infos = [vk::DescriptorBufferInfo::default()
                    .buffer(self.per_frame_uniform_buf.buf)
                    .offset(0)
                    .range(PER_FRAME_UNIFORM_DATA_SIZE as u64)];
                let desc_writes = [vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .buffer_info(&desc_buf_infos)];
                renderer
                    .vkbase
                    .device
                    .update_descriptor_sets(&desc_writes, &[]);

                desc_set
            }
        };

        self.renderer = Some(renderer);
    }

    fn update(&mut self, camera: &Camera) {
        let renderer = self.renderer.as_ref().unwrap();

        let distance_to_camera = camera.position.length();
        let dist_scale = distance_to_camera / 15.0;
        let scale_matrix = Mat4::from_scale(vec3(dist_scale, dist_scale, dist_scale));

        let image_extent = renderer.vkbase.swapchain.image_extent();
        let image_aspect_ratio = (image_extent.width as f32) / (image_extent.height as f32);
        let pv_matrix = camera.ny_pers_view_matrix(image_aspect_ratio);
        let pvm = pv_matrix * scale_matrix;

        assert!(PER_FRAME_UNIFORM_DATA_SIZE == size_of_var(&pvm));

        let per_frame_uniform_data_offset =
            renderer.in_flight_frame_index() * PER_FRAME_UNIFORM_DATA_SIZE;
        self.per_frame_uniform_buf
            .copy_value(per_frame_uniform_data_offset, &pvm);

        self.update_instance_data_buffer(vec3(0.0, 0.0, 0.0), 0.5);

        self.draw_frame();
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
        unsafe {
            let device = &self.renderer.as_ref().unwrap().vkbase.device;
            device.device_wait_idle().unwrap();
            device.destroy_pipeline(self.pipeline, None);
        }

        {
            let device = &self.renderer.as_ref().unwrap().vkbase.device;
            self.program.destruct(device);
            self.per_frame_uniform_buf.destruct(device);
            self.instance_data_buffer.destruct(device);
            self.arrow.destruct(device);
        }

        self.renderer.as_mut().unwrap().destruct();
    }
}

fn main() {
    let camera = CameraBuilder::new()
        .position(vec3(0.0, 0.0, 15.0))
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
        Box::new(GizmoTest::default()),
        camera,
    );

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let _result = event_loop.run_app(&mut app);
    println!("run_app: {:?}", _result);
}
