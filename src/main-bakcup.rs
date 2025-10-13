mod buffer;
mod camera;
mod common;
mod gui;
mod image;
mod light;
mod mesh;
mod pipeline;
mod renderer;
mod scene;
mod shader;
mod swapchain;
mod texture;
mod util;
mod vkbase;

use ::ash::{Device, vk};
use ::winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use buffer::Buffer;
use camera::{Camera, CameraBuilder};
use glam::{Mat4, Vec3, vec2, vec3};
use gui::{DevGui, Text};
use image::Image;
use mesh::Bounds;
use pipeline::Pipeline;
use renderer::{MAX_FRAMES_IN_FLIGHT, Renderer};
use scene::{Scene, SceneLoader};
use shader::Program;
use std::{f32::consts::PI, rc::Rc, u32, vec::Vec};

const FRAME_UNIFORM_DATA_SIZE: usize = 80;
const SHADOW_FRAME_UNIFORM_DATA_SIZE: usize = 64;

#[derive(Default)]
struct App {
    window_width: u32,
    window_height: u32,
    window: Option<Window>,

    renderer: Option<Renderer>,

    default_program: Program,
    shadow_program: Program,
    dev_gui_program: Program,

    frame_uniform_data_buffer: Buffer,

    shadow_frame_uniform_data_buffer: Buffer,
    shadow_image_size: vk::Extent2D,
    shadow_image: Image,

    default_pipeline: Pipeline,
    shadow_pipeline: Pipeline,
    dev_gui_pipeline: Pipeline,

    dev_gui: DevGui,
    scene: Scene,

    pub camera: Camera,

    pub is_right_button_pressed: bool,
}

impl App {
    pub fn with_window_size(mut self, width: u32, height: u32) -> Self {
        self.window_width = width;
        self.window_height = height;
        self
    }

    fn init(&mut self) {
        let renderer = Renderer::new(self.window.as_ref().unwrap());

        let screen_size = vec2(self.window_width as f32, self.window_height as f32);
        self.dev_gui = DevGui::new(screen_size);

        let font_bitmap_bytes = [(
            self.dev_gui.font_bitmap.pixels.as_slice(),
            vk::Extent3D {
                width: self.dev_gui.font_bitmap.width,
                height: self.dev_gui.font_bitmap.height,
                depth: 1,
            },
        )];
        let font_image_index = renderer.image_pool.new_images_from_bytes(
            &font_bitmap_bytes,
            vk::Format::R8_UNORM,
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::R,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::R,
            },
            renderer.cmd_bufs[0],
            renderer.vkbase.present_queue,
        )[0];
        self.dev_gui.setup_font_texture(&renderer.vkbase.device, font_image_index);

        self.default_program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(renderer.shader_set.get("default.vert").unwrap()),
                Rc::clone(renderer.shader_set.get("default.frag").unwrap()),
            ],
        );
        self.shadow_program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![Rc::clone(renderer.shader_set.get("shadow.vert").unwrap())],
        );
        self.dev_gui_program = Program::new(
            &renderer.vkbase.device,
            vk::PipelineBindPoint::GRAPHICS,
            vec![
                Rc::clone(renderer.shader_set.get("devgui-text.vert").unwrap()),
                Rc::clone(renderer.shader_set.get("devgui-text.frag").unwrap()),
            ],
        );

        assert!(FRAME_UNIFORM_DATA_SIZE % 16 == 0);
        let frame_data_uniform_buffer_size = FRAME_UNIFORM_DATA_SIZE * MAX_FRAMES_IN_FLIGHT;
        self.frame_uniform_data_buffer = Buffer::new(
            &renderer.vkbase.device,
            frame_data_uniform_buffer_size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &renderer.vkbase.device_memory_properties,
        );

        assert!(SHADOW_FRAME_UNIFORM_DATA_SIZE % 16 == 0);
        let shadow_frame_uniform_data_buffer_size =
            SHADOW_FRAME_UNIFORM_DATA_SIZE * MAX_FRAMES_IN_FLIGHT;
        self.shadow_frame_uniform_data_buffer = Buffer::new(
            &renderer.vkbase.device,
            shadow_frame_uniform_data_buffer_size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &renderer.vkbase.device_memory_properties,
        );

        self.shadow_image_size = vk::Extent2D {
            width: 2048,
            height: 2048,
        };
        self.shadow_image = Image::new_depth_image(
            &renderer.vkbase.device,
            &renderer.vkbase.device_memory_properties,
            self.shadow_image_size,
            renderer.vkbase.depth_format,
        );

        self.dev_gui.text(&Text {
            text: "Hello World!".to_owned(),
            start: vec2(30.0, 30.0),
            height: 100.0,
        });

        self.dev_gui.build_vertex_index_buffer(
            &renderer.vkbase.device,
            &renderer.vkbase.device_memory_properties,
        );

        self.scene = SceneLoader::new(
            &renderer.vkbase.device,
            &renderer.vkbase.device_memory_properties,
            1.0, /* max_sampler_anisotropy */
            renderer.cmd_bufs[0],
            renderer.vkbase.present_queue,
        )
        .load_shadow_test();

        let scene_bounds = {
            let mut bounds = self.scene.bounding_box();
            let min_bounds = Bounds {
                min: -Vec3::ONE,
                max: Vec3::ONE,
            };
            bounds.extend(&min_bounds);
            bounds
        };
        let camera_pos = vec3(
            (scene_bounds.max.x + scene_bounds.min.x) / 2.0,
            (scene_bounds.max.y + scene_bounds.min.y) / 2.0,
            scene_bounds.max.z * 1.3,
        );
        let scene_center = vec3(
            (scene_bounds.max.x + scene_bounds.min.x) / 2.0,
            (scene_bounds.max.y + scene_bounds.min.y) / 2.0,
            (scene_bounds.max.z + scene_bounds.min.z) / 2.0,
        );
        self.camera = CameraBuilder::new()
            .position(camera_pos)
            .up(vec3(0.0, 1.0, 0.0))
            .lookat(scene_center - camera_pos)
            .fov_y(40.0 / 180.0 * std::f32::consts::PI)
            .near_z(0.1)
            .build()
            .unwrap();

        let color_attachment_formats = [renderer.vkbase.surface_format.format];

        self.default_pipeline = Pipeline::new_default_graphics_pipeline(
            &renderer.vkbase.device,
            renderer.desc_pool,
            &color_attachment_formats,
            renderer.vkbase.depth_format,
            &self.default_program,
            self.scene.textures.len(),
        );
        self.shadow_pipeline = Pipeline::new_shadow_graphics_pipeline(
            &renderer.vkbase.device,
            renderer.desc_pool,
            renderer.vkbase.depth_format,
            &self.shadow_program,
        );
        self.dev_gui_pipeline = Pipeline::new_dev_gui_graphics_pipeline(
            &renderer.vkbase.device,
            renderer.desc_pool,
            &color_attachment_formats,
            renderer.vkbase.depth_format,
            &self.dev_gui_program,
        );

        self.write_default_pipeline_desc_sets(&renderer.vkbase.device);
        self.write_shadow_pipeline_desc_sets(&renderer.vkbase.device);
        self.write_dev_gui_pipeline_desc_sets(&renderer.vkbase.device);

        self.renderer = Some(renderer);
    }

    fn write_default_pipeline_desc_sets(&self, device: &Device) {
        let desc_buf_info = vk::DescriptorBufferInfo::default()
            .buffer(self.frame_uniform_data_buffer.buf)
            .offset(0)
            .range(FRAME_UNIFORM_DATA_SIZE as u64);

        let mut desc_writes = Vec::with_capacity(1 + self.scene.textures.len());
        desc_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .dst_set(self.default_pipeline.sets[0])
                .dst_binding(0)
                .dst_array_element(0)
                .buffer_info(std::slice::from_ref(&desc_buf_info)),
        );

        let desc_image_infos: Vec<_> = self
            .scene
            .textures
            .iter()
            .map(|texture| {
                vk::DescriptorImageInfo::default()
                    .sampler(texture.sampler)
                    .image_view(texture.image.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();

        for (i, _texture) in self.scene.textures.iter().enumerate() {
            desc_writes.push(
                vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(self.default_pipeline.sets[i + 1])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .image_info(std::slice::from_ref(&desc_image_infos[i])),
            );
        }

        unsafe {
            device.update_descriptor_sets(&desc_writes, &[]);
        }
    }

    fn write_shadow_pipeline_desc_sets(&self, device: &Device) {
        let desc_buf_info = vk::DescriptorBufferInfo::default()
            .buffer(self.shadow_frame_uniform_data_buffer.buf)
            .offset(0)
            .range(SHADOW_FRAME_UNIFORM_DATA_SIZE as u64);

        let desc_writes = [vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .dst_set(self.shadow_pipeline.sets[0])
            .dst_binding(0)
            .dst_array_element(0)
            .buffer_info(std::slice::from_ref(&desc_buf_info))];

        unsafe {
            device.update_descriptor_sets(&desc_writes, &[]);
        }
    }

    fn write_dev_gui_pipeline_desc_sets(&self, device: &Device) {
        let desc_image_info = vk::DescriptorImageInfo::default()
            .sampler(self.dev_gui.textures[0].sampler)
            .image_view(self.dev_gui.textures[0].image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let desc_writes = [vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_set(self.dev_gui_pipeline.sets[0])
            .dst_binding(0)
            .dst_array_element(0)
            .image_info(std::slice::from_ref(&desc_image_info))];

        unsafe {
            device.update_descriptor_sets(&desc_writes, &[]);
        }
    }

    fn destruct(&mut self) {
        if let Some(renderer) = self.renderer.as_mut() {
            let device = &renderer.vkbase.device;

            unsafe {
                device.device_wait_idle().unwrap();
                self.scene.destruct(device);
                self.dev_gui_pipeline.destruct(device);
                self.shadow_pipeline.destruct(device);
                self.default_pipeline.destruct(device);
                self.frame_uniform_data_buffer.destruct(device);
                self.shadow_frame_uniform_data_buffer.destruct(device);
                self.shadow_image.destruct(device);
                self.dev_gui.destruct(device);
                self.default_program.destruct(device);
                self.shadow_program.destruct(device);
                self.dev_gui_program.destruct(device);
                renderer.destruct();
            }
        }
    }

    fn update_frame_uniform_data_buffer(&self, in_flight_frame_index: usize) {
        let vkbase = &self.renderer.as_ref().unwrap().vkbase;

        let image_extent = vkbase.swapchain.image_extent();
        let view_matrix = self.camera.view_matrix();
        let pers_matrix = self
            .camera
            .perspective_matrix((image_extent.width as f32) / (image_extent.height as f32));
        // Compensate for Vulkan NDC's y-axis being pointing downwards.
        let negative_y_matrix = Mat4::from_scale(vec3(1.0, -1.0, 1.0));
        let pv_matrix = [negative_y_matrix * pers_matrix * view_matrix];

        let camera_transform_size = size_of::<Mat4>();

        let frame_uniform_data_offset = in_flight_frame_index * FRAME_UNIFORM_DATA_SIZE;
        self.frame_uniform_data_buffer
            .copy_data(frame_uniform_data_offset, &pv_matrix);

        let light_data = [self.scene.directional_light.direction.normalize()];
        self.frame_uniform_data_buffer.copy_data(
            frame_uniform_data_offset + camera_transform_size,
            &light_data,
        );
    }

    fn update_shadow_frame_uniform_data_buffer(&self, in_flight_frame_index: usize) {
        let light_proj_matrix = [self.scene.directional_light.projection_matrix()];

        let offset = in_flight_frame_index * SHADOW_FRAME_UNIFORM_DATA_SIZE;
        self.shadow_frame_uniform_data_buffer
            .copy_data(offset, &light_proj_matrix);
    }

    fn render_shadow_pipeline(
        &self,
        device: &Device,
        cmd_buf: vk::CommandBuffer,
        in_flight_frame_index: usize,
    ) {
        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.shadow_image.view)
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
                extent: self.shadow_image_size,
            })
            .layer_count(1)
            .depth_attachment(&depth_attachment_info);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.shadow_image_size.width as f32,
            height: self.shadow_image_size.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor: vk::Rect2D = self.shadow_image_size.into();

        unsafe {
            device.cmd_begin_rendering(cmd_buf, &rendering_info);

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline.pipeline,
            );

            device.cmd_set_viewport(cmd_buf, 0, &[viewport]);
            device.cmd_set_scissor(cmd_buf, 0, &[scissor]);

            for entity in &self.scene.entities {
                let mesh = &self.scene.meshes[entity.mesh_index as usize];
                device.cmd_bind_vertex_buffers(cmd_buf, 0, &[mesh.vertex_buffer.buf], &[0]);

                device.cmd_bind_index_buffer(
                    cmd_buf,
                    mesh.index_buffer.buf,
                    0,
                    vk::IndexType::UINT32,
                );

                device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    self.shadow_program.bind_point,
                    self.shadow_program.pipeline_layout,
                    0,
                    &self.shadow_pipeline.sets[0..1],
                    &[(in_flight_frame_index * SHADOW_FRAME_UNIFORM_DATA_SIZE) as u32],
                );

                for submesh in &mesh.submeshes {
                    device.cmd_draw_indexed(
                        cmd_buf,
                        submesh.index_count,
                        1,
                        submesh.index_offset,
                        submesh.vertex_offset,
                        1,
                    );
                }
            }

            device.cmd_end_rendering(cmd_buf);
        }
    }

    fn render_default_pipeline(
        &self,
        cmd_buf: vk::CommandBuffer,
        present_image_view: vk::ImageView,
        in_flight_frame_index: usize,
    ) {
        let vkbase = &self.renderer.as_ref().unwrap().vkbase;

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
            .image_view(vkbase.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });

        let image_extent = vkbase.swapchain.image_extent();

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: image_extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info))
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
            vkbase.device.cmd_begin_rendering(cmd_buf, &rendering_info);

            vkbase.device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.default_pipeline.pipeline,
            );

            vkbase.device.cmd_set_viewport(cmd_buf, 0, &[viewport]);
            vkbase.device.cmd_set_scissor(cmd_buf, 0, &[scissor]);

            for entity in &self.scene.entities {
                let mesh = &self.scene.meshes[entity.mesh_index as usize];
                vkbase
                    .device
                    .cmd_bind_vertex_buffers(cmd_buf, 0, &[mesh.vertex_buffer.buf], &[0]);

                vkbase.device.cmd_bind_index_buffer(
                    cmd_buf,
                    mesh.index_buffer.buf,
                    0,
                    vk::IndexType::UINT32,
                );

                vkbase.device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    self.default_program.bind_point,
                    self.default_program.pipeline_layout,
                    0,
                    &self.default_pipeline.sets[0..1],
                    &[(in_flight_frame_index * FRAME_UNIFORM_DATA_SIZE) as u32],
                );

                for (submesh, texture_i) in mesh.submeshes.iter().zip(entity.texture_indices.iter())
                {
                    let texture_i = *texture_i as usize;
                    vkbase.device.cmd_bind_descriptor_sets(
                        cmd_buf,
                        self.default_program.bind_point,
                        self.default_program.pipeline_layout,
                        1,
                        &self.default_pipeline.sets[(texture_i + 1)..(texture_i + 2)],
                        &[],
                    );

                    vkbase.device.cmd_draw_indexed(
                        cmd_buf,
                        submesh.index_count,
                        1,
                        submesh.index_offset,
                        submesh.vertex_offset,
                        1,
                    );
                }
            }

            vkbase.device.cmd_end_rendering(cmd_buf);
        }
    }

    fn render_dev_gui_pipeline(
        &self,
        cmd_buf: vk::CommandBuffer,
        present_image_view: vk::ImageView,
    ) {
        let vkbase = &self.renderer.as_ref().unwrap().vkbase;

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(present_image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vkbase.swapchain.image_extent(),
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info));

        let image_extent = vkbase.swapchain.image_extent();
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
            vkbase.device.cmd_begin_rendering(cmd_buf, &rendering_info);

            vkbase.device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.dev_gui_pipeline.pipeline,
            );

            vkbase.device.cmd_set_viewport(cmd_buf, 0, &[viewport]);
            vkbase.device.cmd_set_scissor(cmd_buf, 0, &[scissor]);

            vkbase.device.cmd_bind_vertex_buffers(
                cmd_buf,
                0,
                &[self.dev_gui.vertex_buffer.buf],
                &[0],
            );

            vkbase.device.cmd_bind_index_buffer(
                cmd_buf,
                self.dev_gui.index_buffer.buf,
                0,
                vk::IndexType::UINT32,
            );

            vkbase.device.cmd_bind_descriptor_sets(
                cmd_buf,
                self.dev_gui_program.bind_point,
                self.dev_gui_program.pipeline_layout,
                0,
                &self.dev_gui_pipeline.sets[0..1],
                &[],
            );

            vkbase
                .device
                .cmd_draw_indexed(cmd_buf, self.dev_gui.indices.len() as u32, 1, 0, 0, 1);

            vkbase.device.cmd_end_rendering(cmd_buf);
        }
    }

    pub fn update(&mut self) {
        let frame_index = self.renderer.as_ref().unwrap().begin_frame();

        let present_image_view = self.renderer.as_ref().unwrap().vkbase.present_image_views
            [frame_index.present_image_index];
        let cmd_buf = self.renderer.as_ref().unwrap().cmd_bufs[frame_index.in_flight_frame_index];

        self.update_frame_uniform_data_buffer(frame_index.in_flight_frame_index);
        self.update_shadow_frame_uniform_data_buffer(frame_index.in_flight_frame_index);

        self.render_shadow_pipeline(
            &self.renderer.as_ref().unwrap().vkbase.device,
            cmd_buf,
            frame_index.in_flight_frame_index,
        );
        self.render_default_pipeline(
            cmd_buf,
            present_image_view,
            frame_index.in_flight_frame_index,
        );
        self.render_dev_gui_pipeline(cmd_buf, present_image_view);

        self.renderer.as_mut().unwrap().end_frame(&frame_index);
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

        self.init();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let scale = 0.52;
                if event.state.is_pressed() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::ArrowLeft)
                        | PhysicalKey::Code(KeyCode::KeyA) => {
                            self.camera.translate_local(vec3(-scale, 0.0, 0.0));
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight)
                        | PhysicalKey::Code(KeyCode::KeyD) => {
                            self.camera.translate_local(vec3(scale, 0.0, 0.0));
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                            self.camera.translate_local(vec3(0.0, 0.0, -scale));
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown)
                        | PhysicalKey::Code(KeyCode::KeyS) => {
                            self.camera.translate_local(vec3(0.0, 0.0, scale));
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    match state {
                        ElementState::Pressed => self.is_right_button_pressed = true,
                        ElementState::Released => self.is_right_button_pressed = false,
                    }
                }
            }
            WindowEvent::CloseRequested => {
                println!("[DEBUG LINW] close requested.");
                self.destruct();
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.update();
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                println!(
                    "[DEBUG LINW] resized requested: (w: {}, h: {})",
                    size.width, size.height
                );
                if let Some(renderer) = self.renderer.as_mut() {
                    let recreated = renderer.vkbase.recreate_swapchain(vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    });

                    if recreated {
                        renderer.update_depth_image();
                    }
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
                if self.is_right_button_pressed {
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
