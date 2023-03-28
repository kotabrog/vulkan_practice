use nalgebra_glm as glm;
use winit::event::{ElementState, VirtualKeyCode, KeyboardInput};

use crate::app::App;

#[allow(dead_code)]
pub fn models_up_down(app: &mut App, input: KeyboardInput) {
    if input.state == ElementState::Pressed {
        match input.virtual_keycode {
            Some(VirtualKeyCode::Left) if app.models > 1 => app.models -= 1,
            Some(VirtualKeyCode::Right) if app.models < 4 => app.models += 1,
            _ => { }
        }
    }
}

#[allow(dead_code)]
pub fn translate_key(app: &mut App, input: KeyboardInput) {
    if input.state == ElementState::Pressed {
        match input.virtual_keycode {
            Some(VirtualKeyCode::Left) => app.translate(glm::vec3(0.0, -0.1, 0.0)),
            Some(VirtualKeyCode::Right) => app.translate(glm::vec3(0.0, 0.1, 0.0)),
            Some(VirtualKeyCode::Up) => app.translate(glm::vec3(0.0, 0.0, 0.1)),
            Some(VirtualKeyCode::Down) => app.translate(glm::vec3(0.0, 0.0, -0.1)),
            _ => { }
        }
    }
}
