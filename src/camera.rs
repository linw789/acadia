use glam::{Mat3, Mat4, Vec3};
use std::f32::consts::PI;

pub struct Camera {
    position: Vec3,
    rotation: Mat3,
    rotation_world_y: f32,
    rotation_local_x: f32,
    fov_y: f32,
    near_z: f32,
}

impl Camera {
    /// `fov_y` is the field-of-view along y-axis in radian.
    pub fn new(pos: Vec3, fov_y: f32, near_z: f32) -> Self {
        Self {
            position: pos,
            fov_y,
            near_z,
            ..Default::default()
        }
    }

    pub fn lookat_dir(&self) -> Vec3 {
        self.rotation.z_axis
    }

    pub fn translate_local(&mut self, t: Vec3) {
        self.position +=
            t.x * self.rotation.x_axis + t.y * self.rotation.y_axis + t.z * self.rotation.z_axis;
    }

    /// Rotate camera around its world y-axis by `angle` in radian.
    pub fn rotate_world_y(&mut self, angle: f32) {
        self.rotation_world_y += angle;
        let m3_world_y = Mat3::from_rotation_y(self.rotation_world_y);
        let m3_local_x = Mat3::from_axis_angle(m3_world_y.x_axis, self.rotation_local_x);
        self.rotation = m3_local_x * m3_world_y;
    }

    /// Rotate camera around its local x-axis by `angle` in radian. The total degree of rotation is
    /// clamped to [-PI/2, PI/2].
    pub fn rotate_local_x(&mut self, angle: f32) {
        let rx = self.rotation_local_x + angle;
        self.rotation_local_x = f32::max(-0.5 * PI, f32::min(0.5 * PI, rx));
        let m3_world_y = Mat3::from_rotation_y(self.rotation_world_y);
        let m3_local_x = Mat3::from_axis_angle(m3_world_y.x_axis, self.rotation_local_x);
        self.rotation = m3_local_x * m3_world_y;
    }

    /// Return the matrix that transforms a point from world space into the camera space.
    pub fn view_matrix(&self) -> Mat4 {
        let r = Mat4::from_mat3(self.rotation.inverse());
        let t = Mat4::from_translation(-self.position);
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
            position: Vec3::ZERO,
            rotation: Mat3::IDENTITY,
            rotation_world_y: 0.0,
            rotation_local_x: 0.0,
            fov_y: 45.0,
            near_z: 0.0,
        }
    }
}
