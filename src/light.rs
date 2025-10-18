use glam::{Mat4, Vec2, Vec3, vec3, vec4};

pub struct DirectionalLight {
    pub position: Vec3,
    pub direction: Vec3,
    up: Vec3,
}

impl DirectionalLight {
    pub fn new(position: Vec3, direction: Vec3, up: Vec3) -> Self {
        Self {
            position,
            direction,
            up,
        }
    }

    pub fn projection_matrix(&self, view_size: Vec2) -> Mat4 {
        let up = self.up.normalize();
        let z_axis = (-self.direction).normalize();
        assert!(f32::abs(Vec3::dot(up, z_axis)) < 0.99);

        let x_axis = Vec3::cross(up, z_axis).normalize();
        let y_axis = Vec3::cross(z_axis, x_axis).normalize();

        let base_matrix = Mat4::from_cols(
            (x_axis, 0.0).into(),
            (y_axis, 0.0).into(),
            (z_axis, 0.0).into(),
            vec4(0.0, 0.0, 0.0, 1.0),
        );
        let view_matrix = base_matrix.inverse();

        let orthographic_matrix = {
            let width = view_size.x;
            let height = view_size.y;
            let nearz = -0.1;
            let farz = -59.0;

            Mat4::from_cols(
                vec4(2.0 / width, 0.0, 0.0, 0.0),
                vec4(0.0, -2.0 / height, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0 / (nearz - farz), 0.0),
                vec4(0.0, 0.0, -farz / (nearz - farz), 1.0),
            )
        };

        orthographic_matrix * view_matrix
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            position: vec3(0.0, 0.0, 0.0),
            direction: vec3(1.0, 0.0, 0.0),
            up: vec3(0.0, 1.0, 0.0),
        }
    }
}
