use crate::{camera::Camera, input::MouseState};
use ::glam::Vec2;
use ::winit::{dpi::PhysicalSize, window::Window};

pub trait Scene {
    fn init(&mut self, window: &Window);
    fn update(&mut self, camera: &Camera, mouse_state: &MouseState);
    fn resize(&mut self, new_size: PhysicalSize<u32>);
    fn destruct(&mut self);
}
