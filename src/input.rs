use glam::Vec2;

pub struct MouseState {
    pub right_button_pressed: bool,
    pub left_button_pressed: bool,
    pub cursor_position: Vec2,
    pub cursor_delta: Vec2,
}
