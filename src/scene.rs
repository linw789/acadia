use crate::camera::Camera;
use ::winit::{dpi::PhysicalSize, window::Window};

pub trait Scene {
    fn init(&mut self, window: &Window);
    fn update(&mut self, camera: &Camera);
    fn resize(&mut self, new_size: PhysicalSize<u32>);
    fn destruct(&mut self);
}
