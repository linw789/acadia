use glam::{Mat4, Vec3, vec4};

pub struct DirectionalLight {
    pub position: Vec3,
    pub direction: Vec3,
}

impl DirectionalLight {
    pub fn new(position: Vec3, direction: Vec3) -> Self {
        Self {
            position,
            direction,
        }
    }

    pub fn projection_matrix(&self) -> Mat4 {
        let width = 40.0;
        let height = 40.0;
        let nearz = 0.1;
        let farz = 59.0;

        let m = Mat4::from_cols(
            vec4(1.0 / width, 0.0, 0.0, 0.0),
            vec4(0.0, 1.0 / height, 0.0, 0.0),
            vec4(0.0, 0.0, -1.0 / (farz - nearz), 0.0),
            vec4(0.0, 0.0, farz / (farz - nearz), 1.0),
        );

        m.inverse()
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        }
    }
}
