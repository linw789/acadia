use crate::common::Transform;
use glam::{Mat4, Quat, Vec3};

pub struct Camera {
    pub transform: Transform,
    fov_y: f32,
    near_z: f32,
}

impl Camera {
    /// `fov_y` is the field-of-view along y-axis in radian.
    pub fn new(pos: Vec3, fov_y: f32, near_z: f32) -> Self {
        Self {
            transform: Transform {
                position: pos,
                orientation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            fov_y,
            near_z,
        }
    }

    /// Return the matrix that transforms a point from world space into the camera space.
    pub fn view_matrix(&self) -> Mat4 {
        let r = Mat4::from_quat(self.transform.orientation.inverse());
        let t = Mat4::from_translation(-self.transform.position);
        r * t
    }

    /// Return the perspective matrix with z_near and infinity mapped to [0, 1].
    /// `aspect_ratio` is the ratio of viewport's width over height.
    pub fn perspective_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_infinite_rh(self.fov_y, aspect_ratio, self.near_z)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            transform: Transform {
                position: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            fov_y: 45.0,
            near_z: 0.0,
        }
    }
}
