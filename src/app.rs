use crate::{camera::Camera, scene::Scene, input::MouseState};
use ::winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use glam::{Vec2, vec2, vec3};
use std::f32::consts::PI;

pub struct App {
    window_size: PhysicalSize<u32>,
    window: Option<Window>,

    scene: Box<dyn Scene>,
    camera: Camera,

    exit_requested: bool,
    is_left_button_pressed: bool,
    is_right_button_pressed: bool,
    cursor_pos: Vec2,
    cursor_delta: Vec2,
}

impl App {
    pub fn new(window_size: PhysicalSize<u32>, scene: Box<dyn Scene>, camera: Camera) -> Self {
        Self {
            window_size,
            window: None,
            scene,
            camera,
            exit_requested: false,
            is_left_button_pressed: false,
            is_right_button_pressed: false,
            cursor_pos: Vec2::ZERO,
            cursor_delta: Vec2::ZERO,
        }
    }

    fn init(&mut self) {
        self.scene.init(self.window.as_ref().unwrap());
    }

    fn update(&mut self) {
        let mouse_state = MouseState {
            right_button_pressed: self.is_right_button_pressed,
            left_button_pressed: self.is_left_button_pressed,
            cursor_position: self.cursor_pos,
            cursor_delta: self.cursor_delta,
        };
        self.scene.update(&self.camera, &mouse_state);
        self.cursor_delta = Vec2::ZERO;
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.scene.resize(new_size);
    }

    fn destruct(&mut self) {
        self.scene.destruct();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(self.window_size)
                        .with_title("Acadia"),
                )
                .unwrap(),
        );

        self.init();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let scale = 0.1;
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
                if button == MouseButton::Left {
                    match state {
                        ElementState::Pressed => self.is_left_button_pressed = true,
                        ElementState::Released => self.is_left_button_pressed = false,
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = vec2(position.x as f32, position.y as f32);
            }
            WindowEvent::CloseRequested => {
                println!("[DEBUG LINW] close requested.");
                self.destruct();
                event_loop.exit();
                self.exit_requested = true;
            }
            WindowEvent::RedrawRequested => {
                if self.exit_requested == false {
                    self.update();
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                println!(
                    "[DEBUG LINW] resized requested: (w: {}, h: {})",
                    size.width, size.height
                );
                self.resize(size);
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
                self.cursor_delta = vec2(delta.0 as f32, delta.1 as f32);

                if self.is_right_button_pressed {
                    let scale = 0.2;
                    let rx = scale * self.cursor_delta.x / 180.0 * PI;
                    let ry = scale * self.cursor_delta.y / 180.0 * PI;

                    self.camera.rotate_world_y(-rx);
                    self.camera.rotate_local_x(-ry);
                }
            }
            _ => (),
        }
    }
}
