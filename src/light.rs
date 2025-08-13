use glam::Vec3;

pub struct DirectionalLight {
    pub direction: Vec3,
}

impl DirectionalLight {
    pub fn new(direction: Vec3) -> Self {
        Self { direction }
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(1.0, 0.0, 0.0),
        }
    }
}
