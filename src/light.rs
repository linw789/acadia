use crate::mesh::Aabb;
use glam::{Mat3, Mat4, Vec3, Vec4, vec3, vec4};

pub struct DirectionalLight {
    pub position: Vec3,
    bases: Mat3,
}

impl DirectionalLight {
    pub fn new(position: Vec3, direction: Vec3, up: Vec3) -> Self {
        let up = up.normalize();
        let z_axis = (-direction).normalize();
        assert!(f32::abs(Vec3::dot(up, z_axis)) < 0.99);

        let x_axis = Vec3::cross(up, z_axis).normalize();
        let y_axis = Vec3::cross(z_axis, x_axis).normalize();

        let bases = Mat3::from_cols(x_axis, y_axis, z_axis);

        Self { position, bases }
    }

    pub fn direction(&self) -> Vec3 {
        -self.bases.z_axis
    }

    pub fn view_matrix(&self) -> Mat4 {
        let v = Mat4::from_translation(self.position) * Mat4::from_mat3(self.bases);
        v.inverse()
    }

    pub fn ny_orthographic_projection(&self, scene_bounds: &Aabb) -> Mat4 {
        let view_matrix = self.view_matrix();

        // Note, to calculate the bounding box of scene_bounds in the light space, we cannot just
        // check the min and max of scene_bounds. We need to check each of the eight corners of the
        // bounding box.

        let bounding_points = [
            // bottom rectangle
            Vec4::from((scene_bounds.min, 1.0)),
            vec4(
                scene_bounds.min.x,
                scene_bounds.min.y,
                scene_bounds.max.z,
                1.0,
            ),
            vec4(
                scene_bounds.max.x,
                scene_bounds.min.y,
                scene_bounds.max.z,
                1.0,
            ),
            vec4(
                scene_bounds.max.x,
                scene_bounds.min.y,
                scene_bounds.min.z,
                1.0,
            ),
            // top rectable
            Vec4::from((scene_bounds.max, 1.0)),
            vec4(
                scene_bounds.max.x,
                scene_bounds.max.y,
                scene_bounds.min.z,
                1.0,
            ),
            vec4(
                scene_bounds.min.x,
                scene_bounds.max.y,
                scene_bounds.min.z,
                1.0,
            ),
            vec4(
                scene_bounds.min.x,
                scene_bounds.max.y,
                scene_bounds.max.z,
                1.0,
            ),
        ];

        let mut bounds_light_space = Aabb {
            min: Vec3::MAX,
            max: Vec3::MIN,
        };
        for point in &bounding_points {
            // Transform bounding points into the light space.
            let p4 = view_matrix * point;
            let p3 = vec3(p4.x, p4.y, p4.z);
            bounds_light_space.min = bounds_light_space.min.min(p3);
            bounds_light_space.max = bounds_light_space.max.max(p3);
        }

        let neg_y_orthographic = {
            let margin = 1.0;
            let left = bounds_light_space.min.x;
            let right = bounds_light_space.max.x;
            let bottom = bounds_light_space.min.y;
            let top = bounds_light_space.max.y;
            let nearz = bounds_light_space.max.z + margin;
            let farz = bounds_light_space.min.z - margin;

            Mat4::from_cols(
                vec4(2.0 / (right - left), 0.0, 0.0, 0.0),
                vec4(0.0, 2.0 / (bottom - top), 0.0, 0.0),
                vec4(0.0, 0.0, 1.0 / (nearz - farz), 0.0),
                vec4(
                    -(right + left) / (right - left),
                    -(bottom + top) / (bottom - top),
                    farz / (farz - nearz),
                    1.0,
                ),
            )
        };

        neg_y_orthographic * view_matrix

        // let negative_y_matrix = Mat4::from_scale(vec3(1.0, -1.0, 1.0));
        // let pers_matrix = Mat4::perspective_infinite_reverse_rh(
        //     40.0 / 180.0 * std::f32::consts::PI,
        //     1920.0/1080.0,
        //     0.1);
        // negative_y_matrix * pers_matrix * view_matrix
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            position: vec3(0.0, 0.0, 0.0),
            bases: Mat3::IDENTITY,
        }
    }
}
