use glam::{Mat3, Mat4, Quat, Vec3};

#[derive(Clone, Debug, Default, Copy)]
#[repr(C, packed)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, Default, Copy)]
#[repr(C, packed)]
pub struct Vertex2D {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, Default, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub orientation: Quat,
    pub scale: Vec3,
}

impl Transform {
    /// Translate in the local space. The three components of `t` represents translation along the
    /// local x-axis, y-axis and z-axis respectively.
    pub fn translate_local(&mut self, t: Vec3) {
        let rotation_matrix = Mat3::from_quat(self.orientation);
        self.position += t.x * rotation_matrix.x_axis;
        self.position += t.y * rotation_matrix.y_axis;
        self.position += t.z * rotation_matrix.z_axis;
    }

    /// Rotate around the local x-axis. `angle` is the rotating angle in radian.
    pub fn rotate_local_x(&mut self, angle: f32) {
        let rotation_matrix = Mat3::from_quat(self.orientation);
        let r = Quat::from_axis_angle(rotation_matrix.x_axis, angle);
        self.orientation = r * self.orientation;
        self.orientation = self.orientation.normalize();
    }

    /// Rotate around the local y-axis. `angle` is the rotating angle in radian.
    pub fn rotate_local_y(&mut self, angle: f32) {
        let rotation_matrix = Mat3::from_quat(self.orientation);
        let r = Quat::from_axis_angle(rotation_matrix.y_axis, angle);
        self.orientation = r * self.orientation;
        self.orientation = self.orientation.normalize();
    }

    /// Rotate around the local z-axis. `angle` is the rotating angle in radian.
    pub fn rotate_local_z(&mut self, angle: f32) {
        let rotation_matrix = Mat3::from_quat(self.orientation);
        let r = Quat::from_axis_angle(rotation_matrix.z_axis, angle);
        self.orientation = r * self.orientation;
        self.orientation = self.orientation.normalize();
    }

    pub fn mat4(&self) -> Mat4 {
        let s = Mat4::from_scale(self.scale);
        let r = Mat4::from_quat(self.orientation);
        let t = Mat4::from_translation(self.position);
        t * r * s
    }
}
