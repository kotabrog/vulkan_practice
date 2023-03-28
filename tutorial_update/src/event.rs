use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Window;
use winit::dpi::PhysicalSize;

use crate::app::App;
// use keyboard::models_up_down;
use keyboard::translate_key;

mod keyboard;

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
            Event::MainEventsCleared if !self.destroying && !self.minimized =>
                self.render(app, window),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } =>
                self.resize(size, app),
            // Destroy our Vulkan app.
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } =>
                self.close(app, control_flow),
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } =>
                translate_key(app, input),
                // models_up_down(app, input),
            _ => {}
        }
    }

    fn render(&self, app: &mut App, window: &Window) {
        unsafe { app.render(&window) }.unwrap();
    }

    fn resize(&mut self, size: PhysicalSize<u32>, app: &mut App) {
        if size.width == 0 || size.height == 0 {
            self.minimized = true;
        } else {
            self.minimized = false;
            app.resized = true;
        }
    }

    fn close(&mut self, app: &mut App, control_flow: &mut ControlFlow) {
        self.destroying = true;
        *control_flow = ControlFlow::Exit;
        unsafe { app.destroy(); }
    }
}
