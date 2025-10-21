use glam::{Mat3, Mat4, Vec3, vec3};
use std::{
    f32::consts::PI,
    io::{Error, ErrorKind},
    result::Result,
};

pub struct Camera {
    position: Vec3,
    bases: Mat3,
    rotation_world_y: f32,
    rotation_local_x: f32,
    fov_y: f32,
    near_z: f32,
}

pub struct CameraBuilder {
    position: Vec3,
    up: Vec3,
    lookat: Vec3,
    fov_y: f32,
    near_z: f32,
}

impl CameraBuilder {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            up: Vec3::new(0.0, 1.0, 0.0),
            lookat: Vec3::new(0.0, 0.0, -1.0),
            fov_y: 40.0,
            near_z: -0.5,
        }
    }

    pub fn position(mut self, pos: Vec3) -> Self {
        self.position = pos;
        self
    }

    /// `up` is a normalized direction pointing upwards from the camera. It must not be parallel
    /// with the `lookat` direction.
    pub fn up(mut self, up: Vec3) -> Self {
        self.up = up;
        self
    }

    /// `lookat` is the normalized direction the camera is viewing. It must not be parallel with
    /// the `up` direction.
    pub fn lookat(mut self, lookat: Vec3) -> Self {
        self.lookat = lookat;
        self
    }

    /// `fov_y` is the field-of-view along y-axis in radian.
    pub fn fov_y(mut self, fov_y: f32) -> Self {
        self.fov_y = fov_y;
        self
    }

    pub fn near_z(mut self, near_z: f32) -> Self {
        self.near_z = near_z;
        self
    }

    /// Return a `Camera`. If `up` and `lookat` are almost parallel, return ErrorKind::InvalidInput.
    pub fn build(self) -> Result<Camera, Error> {
        let up = self.up.normalize();
        let lookat_dir = self.lookat - self.position;
        // To maintain the right-hand coordinate system, the camera lookat direction should points
        // towards the camera's negative z-axis.
        let z_axis = (-lookat_dir).normalize();

        if f32::abs(Vec3::dot(up, z_axis)) > 0.99 {
            return Err(Error::from(ErrorKind::InvalidInput));
        }

        let x_axis = Vec3::cross(up, z_axis).normalize();
        let y_axis = Vec3::cross(z_axis, x_axis).normalize();

        Ok(Camera {
            position: self.position,
            bases: Mat3 {
                x_axis,
                y_axis,
                z_axis,
            },
            fov_y: self.fov_y,
            near_z: self.near_z,
            ..Default::default()
        })
    }
}

impl Camera {
    pub fn lookat_dir(&self) -> Vec3 {
        self.bases.z_axis
    }

    pub fn translate_local(&mut self, t: Vec3) {
        self.position +=
            t.x * self.bases.x_axis + t.y * self.bases.y_axis + t.z * self.bases.z_axis;
    }

    /// Rotate camera around its world y-axis by `angle` in radian.
    pub fn rotate_world_y(&mut self, angle: f32) {
        self.rotation_world_y += angle;
        let m3_world_y = Mat3::from_rotation_y(self.rotation_world_y);
        let m3_local_x = Mat3::from_axis_angle(m3_world_y.x_axis, self.rotation_local_x);
        self.bases = m3_local_x * m3_world_y;
    }

    /// Rotate camera around its local x-axis by `angle` in radian. The total degree of bases is
    /// clamped to [-PI/2, PI/2].
    pub fn rotate_local_x(&mut self, angle: f32) {
        let rx = self.rotation_local_x + angle;
        self.rotation_local_x = f32::max(-0.5 * PI, f32::min(0.5 * PI, rx));
        let m3_world_y = Mat3::from_rotation_y(self.rotation_world_y);
        let m3_local_x = Mat3::from_axis_angle(m3_world_y.x_axis, self.rotation_local_x);
        self.bases = m3_local_x * m3_world_y;
    }

    /// Return the matrix that transforms a point from world space into the camera space.
    pub fn view_matrix(&self) -> Mat4 {
        // let r = Mat4::from_mat3(self.bases.inverse());
        // let t = Mat4::from_translation(-self.position);
        // r * t
        // Is Mat3::inverse() faster than Mat4::inverse()?
        let v = Mat4::from_translation(self.position) * Mat4::from_mat3(self.bases);
        v.inverse()
    }

    /// Return the perspective matrix with z_near and infinity mapped to [1, 0].
    /// `aspect_ratio` is the ratio of viewport's width over height.
    pub fn perspective_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect_ratio, self.near_z)
    }

    /// Return negative_y_matrix * perspective_matrix * view_matrix.
    pub fn ny_pers_view_matrix(&self, aspect_ratio: f32) -> Mat4 {
        let view_matrix = self.view_matrix();
        let pers_matrix = self.perspective_matrix(aspect_ratio);
        // Compensate for Vulkan NDC's y-axis being pointing downwards.
        let negative_y_matrix = Mat4::from_scale(vec3(1.0, -1.0, 1.0));
        negative_y_matrix * pers_matrix * view_matrix
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            bases: Mat3::IDENTITY,
            rotation_world_y: 0.0,
            rotation_local_x: 0.0,
            fov_y: 45.0,
            near_z: 0.0,
        }
    }
}
