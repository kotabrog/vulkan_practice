use winit::event::{Event, WindowEvent, ElementState, VirtualKeyCode};
use winit::event_loop::ControlFlow;
use winit::window::Window;

use crate::app::App;

pub struct EventHandler {
    destroying: bool,
    minimized: bool,
}

impl EventHandler {
    pub fn new() -> Self {
        Self {
            destroying: false,
            minimized: false,
        }
    }

    pub fn run(&mut self, app: &mut App, window: &Window, event: Event<()>, control_flow: &mut ControlFlow) {
        *control_flow = ControlFlow::Poll;
        match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::MainEventsCleared if !self.destroying && !self.minimized  =>
                unsafe { app.render(&window) }.unwrap(),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width == 0 || size.height == 0 {
                    self.minimized = true;
                } else {
                    self.minimized = false;
                    app.resized = true;
                }
            }
            // Destroy our Vulkan app.
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                self.destroying = true;
                *control_flow = ControlFlow::Exit;
                unsafe { app.destroy(); }
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if input.state == ElementState::Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::Left) if app.models > 1 => app.models -= 1,
                        Some(VirtualKeyCode::Right) if app.models < 4 => app.models += 1,
                        _ => { }
                    }
                }
            }
            _ => {}
        }
    }
}
