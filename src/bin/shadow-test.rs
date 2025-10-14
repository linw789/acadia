use ::ash::vk;
use ::glam::{Mat4, Vec3, vec3};
use ::winit::{
    dpi::PhysicalSize,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use acadia::{
    app::App,
    buffer::Buffer,
    camera::Camera,
    common::Vertex,
    mesh::Mesh,
    offset_of,
    pipeline::new_graphics_pipeline,
    renderer::{MAX_FRAMES_IN_FLIGHT, Renderer},
    scene::Scene,
    shader::Program,
};
use std::rc::Rc;

const LIGHT_PASS_PER_FRAME_UNIFORM_DATA_SIZE: usize = 80;

#[derive(Default)]
struct ShadowTest {
    renderer: Option<Renderer>,
    shadow_pass_program: Program,
    light_pass_program: Program,
    mesh: Mesh,
    light_dir: Vec3,

    shadow_pipeline: vk::Pipeline,
    shadow_desc_set: vk::DescriptorSet,
    light_pipeline: vk::Pipeline,
    light_desc_set: vk::DescriptorSet,

    light_pass_per_frame_uniform_buf: Buffer,
}

impl ShadowTest {
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
                self.light_pipeline,
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
                &[self.mesh.vertex_buffer.buf],
                &[0],
            );
            renderer.vkbase.device.cmd_bind_index_buffer(
                cmd_buf,
                self.mesh.index_buffer.buf,
                0,
                vk::IndexType::UINT32,
            );

            renderer.vkbase.device.cmd_bind_descriptor_sets(
                cmd_buf,
                self.light_pass_program.bind_point,
                self.light_pass_program.pipeline_layout,
                0,
                &[self.light_desc_set],
                &[(renderer.in_flight_frame_index() * LIGHT_PASS_PER_FRAME_UNIFORM_DATA_SIZE) as u32],
            );

            for submesh in &self.mesh.submeshes {
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

        renderer.end_frame();
    }
}

impl Scene for ShadowTest {
    fn init(&mut self, window: &Window) {
        let renderer = Renderer::new(window);

        self.light_pass_program = Program::new(
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

        self.mesh = Mesh::from_obj(
            &renderer.vkbase.device,
            &renderer.vkbase.device_memory_properties,
            "assets/meshes/shadow-test.obj",
        );

        self.light_dir = vec3(-1.0, -1.0, 0.0);

        self.light_pipeline = {
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

            new_graphics_pipeline(
                &renderer.vkbase.device,
                &vertex_input_state,
                true,
                &[renderer.vkbase.surface_format.format],
                renderer.vkbase.depth_format,
                &color_blend_state,
                &self.light_pass_program,
            )
        };

        self.light_desc_set = {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(self.light_pass_program.desc_set_layouts())
                .descriptor_pool(renderer.desc_pool);

            unsafe {
                renderer
                    .vkbase
                    .device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap()[0]
            }
        };

        self.light_pass_per_frame_uniform_buf = {
            let per_frame_uniform_buf_total_size =
                LIGHT_PASS_PER_FRAME_UNIFORM_DATA_SIZE * MAX_FRAMES_IN_FLIGHT;
            Buffer::new(
                &renderer.vkbase.device,
                per_frame_uniform_buf_total_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                &renderer.vkbase.device_memory_properties,
            )
        };

        unsafe {
            let desc_buf_infos = [vk::DescriptorBufferInfo::default()
                .buffer(self.light_pass_per_frame_uniform_buf.buf)
                .offset(0)
                .range(LIGHT_PASS_PER_FRAME_UNIFORM_DATA_SIZE as u64)];
            let desc_writes = [vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .dst_set(self.light_desc_set)
                .dst_binding(0)
                .dst_array_element(0)
                .buffer_info(&desc_buf_infos)];
            renderer
                .vkbase
                .device
                .update_descriptor_sets(&desc_writes, &[]);
        }

        self.renderer = Some(renderer);
    }

    fn update(&mut self, camera: &Camera) {
        let renderer = self.renderer.as_ref().unwrap();

        let image_extent = renderer.vkbase.swapchain.image_extent();
        let image_aspect_ratio = (image_extent.width as f32) / (image_extent.height as f32);
        let pv_matrix = [camera.ny_pers_view_matrix(image_aspect_ratio)];

        let mut uniform_data_offset =
            renderer.in_flight_frame_index() * LIGHT_PASS_PER_FRAME_UNIFORM_DATA_SIZE;
        self.light_pass_per_frame_uniform_buf
            .copy_value(uniform_data_offset, &pv_matrix);

        uniform_data_offset += size_of::<Mat4>();
        self.light_pass_per_frame_uniform_buf
            .copy_value(uniform_data_offset, &self.light_dir);

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

            device.destroy_pipeline(self.light_pipeline, None);
            device.destroy_pipeline(self.shadow_pipeline, None);
        }

        {
            let device = &self.renderer.as_ref().unwrap().vkbase.device;

            self.shadow_pass_program.destruct(device);
            self.light_pass_program.destruct(device);
            self.light_pass_per_frame_uniform_buf.destruct(device);

            self.mesh.destruct(device);
        }

        self.renderer.as_mut().unwrap().destruct();
    }
}

fn main() {
    let mut app = App::new(
        PhysicalSize::<u32> {
            width: 1920,
            height: 1080,
        },
        Box::new(ShadowTest::default()),
    );

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let _result = event_loop.run_app(&mut app);
    println!("run_app: {:?}", _result);
}
