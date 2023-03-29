use nalgebra_glm as glm;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};

use crate::app::App;

const MOUSE_MOVE_RATE: f32 = 250.0;
const MOUSE_ROTATE_RATE: f32 = 250.0;
const MOUSE_WHEEL_RATE: f32 = 3.0;

#[derive(Debug)]
pub struct MouseState {
    pub right: bool,
    pub left: bool,
}

impl MouseState {
    pub fn new() -> Self {
        Self { right: false, left: false }
    }
}

#[allow(dead_code)]
/// return true when changes
pub fn mouse_click(app: &mut App, state: ElementState, button: MouseButton) -> bool {
    match button {
        MouseButton::Right => {
            match state {
                ElementState::Pressed => {
                    let flag =  !app.mouse_state.right;
                    app.mouse_state.right = true;
                    flag
                },
                ElementState::Released => {
                    let flag =  app.mouse_state.right;
                    app.mouse_state.right = false;
                    flag
                },
            }
        },
        MouseButton::Left => {
            match state {
                ElementState::Pressed => {
                    let flag =  !app.mouse_state.left;
                    app.mouse_state.left = true;
                    flag
                },
                ElementState::Released => {
                    let flag =  app.mouse_state.left;
                    app.mouse_state.left = false;
                    flag
                },
            }
        },
        _ => { false }
    }
}

pub fn mouse_move(app: &mut App, delta: (f64, f64)) {
    match (app.mouse_state.left, app.mouse_state.right) {
        (false, true) => app.translate(glm::vec3(
            0.0,
            delta.0 as f32 / MOUSE_MOVE_RATE,
            -delta.1 as f32 / MOUSE_MOVE_RATE,
        )),
        (true, false) => app.rotate(glm::vec2(
            delta.0 as f32 / MOUSE_ROTATE_RATE,
            delta.1 as f32 / MOUSE_ROTATE_RATE,
        )),
        _ => {},
    }
}

pub fn mouse_wheel(app: &mut App, delta: MouseScrollDelta) {
    let vertical_scroll = match delta {
        winit::event::MouseScrollDelta::LineDelta(_x, y) => y,
        winit::event::MouseScrollDelta::PixelDelta(position) => position.y as f32,
    };
    app.translate(glm::vec3(vertical_scroll / MOUSE_WHEEL_RATE, 0.0, 0.0));
}
