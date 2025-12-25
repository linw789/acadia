use crate::camera::Camera;
use ::winit::{dpi::PhysicalSize, window::Window};
use ::glam::Vec2;

pub trait Scene {
    fn init(&mut self, window: &Window);
    fn update(&mut self, camera: &Camera, cursor_pos: Vec2);
    fn resize(&mut self, new_size: PhysicalSize<u32>);
    fn destruct(&mut self);
}
