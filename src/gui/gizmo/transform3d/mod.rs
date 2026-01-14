mod rotate;
mod translate;

use crate::renderer::Renderer;
use ::ash::Device;
use ::glam::{Mat4, Vec3};
use rotate::GizmoRotate;
use translate::GizmoTranslate;

#[derive(Default)]
pub struct GizmoTransform3D {
    translate: GizmoTranslate,
    rotate: GizmoRotate,
}

pub enum GizmoTransform3DPicked {
    Translate(Vec3),
}

impl GizmoTransform3D {
    pub fn new(renderer: &Renderer) -> Self {
        Self {
            translate: GizmoTranslate::new(renderer),
            rotate: GizmoRotate::new(renderer),
        }
    }

    pub fn update(
        &mut self,
        in_flight_frame_index: usize,
        pers_view_model_matrix: &Mat4,
        camera_pos: Vec3,
    ) {
        self.translate
            .update(in_flight_frame_index, &pers_view_model_matrix);
        self.rotate
            .update(in_flight_frame_index, &pers_view_model_matrix, camera_pos);
    }

    pub fn draw(&self, renderer: &Renderer) {
        self.translate.draw(renderer);
        self.rotate.draw(renderer);
    }

    pub fn destruct(&mut self, device: &Device) {
        self.translate.destruct(device);
        self.rotate.destruct(device);
    }

    pub fn picked(&self, id: u32) -> Option<Vec3> {
        self.translate.picked(id)
    }
}
