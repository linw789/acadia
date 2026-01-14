use ::ash::{Device, vk};
use ::winit::{
    dpi::PhysicalSize,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use acadia::{
    app::App,
    buffer::Buffer,
    camera::{Camera, CameraBuilder},
    common::Vertex,
    gui::gizmo::transform3d::GizmoTransform3D,
    input::MouseState,
    mesh::Mesh,
    offset_of,
    pipeline::PipelineBuilder,
    renderer::{MAX_FRAMES_IN_FLIGHT, Renderer},
    scene::Scene,
    shader::Program,
};
use glam::{Mat4, Vec3, vec3, Vec4, vec2, Vec2};
use std::rc::Rc;

#[derive(Default)]
struct CubePass {
    program: Program,
    mesh: Mesh,
    pipeline: vk::Pipeline,
    desc_sets: Vec<vk::DescriptorSet>,
    uniform_buf: Buffer,
    translate: Vec3,
}

#[derive(Default)]
struct GizmoTest {
    renderer: Option<Renderer>,
    cube_pass: CubePass,
    gizmo_transform3d: GizmoTransform3D,
    is_left_button_pressed: bool,
    translation_gizmo: Option<Vec3>,
}

impl CubePass {
    const PER_FRAME_UNIFORM_DATA_SIZE: usize = 64;

    fn new(renderer: &Renderer) -> Self {
        let program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(renderer.shader_set.get("cube.vert").unwrap()),
                Rc::clone(renderer.shader_set.get("cube.frag").unwrap()),
            ],
            size_of::<Mat4>(),
        );
        let mesh = Mesh::from_obj(
            &renderer.vkbase.device,
            &renderer.vkbase.device_memory_properties,
            "assets/meshes/cube.obj",
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
            mesh,
            pipeline,
            desc_sets,
            uniform_buf,
            translate: Vec3::ZERO,
        }
    }

    fn update(&mut self, in_flight_frame_index: usize, pers_view_matrix: &Mat4) {
        let per_frame_uniform_data_offset =
            in_flight_frame_index * Self::PER_FRAME_UNIFORM_DATA_SIZE;
        self.uniform_buf
            .copy_value(per_frame_uniform_data_offset, pers_view_matrix);
    }

    fn draw(&self, renderer: &Renderer, transform: &Mat4) {
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
                self.program.bind_point,
                self.program.pipeline_layout,
                0,
                &self.desc_sets,
                &[(renderer.in_flight_frame_index() * Self::PER_FRAME_UNIFORM_DATA_SIZE) as u32],
            );

            assert!(size_of::<Mat4>() == (self.program.push_constant_size as usize));
            renderer.vkbase.device.cmd_push_constants(
                cmd_buf,
                self.program.pipeline_layout,
                self.program.push_constant_stages,
                0,
                std::slice::from_raw_parts(
                    (transform as *const Mat4) as *const u8,
                    std::mem::size_of::<Mat4>(),
                ),
            );

            for submesh in &self.mesh.submeshes {
                renderer.vkbase.device.cmd_draw_indexed(
                    cmd_buf,
                    submesh.index_count,
                    1,
                    submesh.index_offset,
                    submesh.vertex_offset,
                    0,
                );
            }

            renderer.vkbase.device.cmd_end_rendering(cmd_buf);
        }
    }

    fn destruct(&mut self, device: &Device) {
        self.uniform_buf.destruct(device);
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
        self.mesh.destruct(device);
        self.program.destruct(device);
    }
}

impl Scene for GizmoTest {
    fn init(&mut self, window: &Window) {
        let renderer = Renderer::new(window);
        self.gizmo_transform3d = GizmoTransform3D::new(&renderer);
        self.cube_pass = CubePass::new(&renderer);
        self.renderer = Some(renderer);
    }

    fn update(&mut self, camera: &Camera, mouse_state: &MouseState) {
        let renderer = self.renderer.as_mut().unwrap();

        let image_extent = renderer.vkbase.swapchain.image_extent();
        let image_aspect_ratio = (image_extent.width as f32) / (image_extent.height as f32);
        let pv_matrix = camera.ny_pers_view_matrix(image_aspect_ratio);

        let transform3d_gizmo_matrix = {
            let distance_to_camera = camera.position.length();
            let dist_scale = distance_to_camera / 15.0;
            let scale_matrix = Mat4::from_scale(vec3(dist_scale, dist_scale, dist_scale));
            let translation_matrix = Mat4::from_translation(self.cube_pass.translate);
            pv_matrix * translation_matrix * scale_matrix
        };
        
        self.cube_pass
            .update(renderer.in_flight_frame_index(), &pv_matrix);
        self.gizmo_transform3d
            .update(renderer.in_flight_frame_index(), &transform3d_gizmo_matrix, camera.position);

        renderer.begin_frame();

        let cube_transform = Mat4::from_translation(self.cube_pass.translate);
        self.cube_pass.draw(&renderer, &cube_transform);
        self.gizmo_transform3d.draw(renderer);

        renderer.copy_obj_ids_from_image_to_buffer();

        renderer.end_frame();

        if self.is_left_button_pressed == false && mouse_state.left_button_pressed == true {
            let obj_id_index = ((mouse_state.cursor_position.y as u32) * image_extent.width)
                + (mouse_state.cursor_position.x as u32);
            let obj_id_buf = renderer.obj_id_buffer();
            let obj_id = obj_id_buf.get::<u32>(obj_id_index as usize);

            self.is_left_button_pressed = true;
            self.translation_gizmo = self.gizmo_transform3d.picked(obj_id);
        } else if self.is_left_button_pressed == true && mouse_state.left_button_pressed == false
        {
            self.is_left_button_pressed = false;
            self.translation_gizmo = None;
        } 

        if let Some(translater) = self.translation_gizmo && mouse_state.cursor_delta != Vec2::ZERO {
            let translater_world = camera.view_matrix() * Vec4::from((translater, 1.0));
            let translater_camera = translater_world.normalize();
            let translater_2d = vec2(translater_camera.x, translater_camera.y);
            let mouse_delta = vec2(mouse_state.cursor_delta.x, -mouse_state.cursor_delta.y) * 0.1;
            println!("[DEBUG LINW] frame count: {}, translater picked: {}, delta: {}", renderer.frame_count(), translater, mouse_delta);
            let delta = Vec2::dot(translater_2d, mouse_delta);
            self.cube_pass.translate += translater * delta;
        }
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
        }

        {
            let device = &self.renderer.as_ref().unwrap().vkbase.device;
            self.cube_pass.destruct(device);
            self.gizmo_transform3d.destruct(device);
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
