#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use anyhow::Result;
// use lazy_static::lazy_static;
use vulkanalia::prelude::v1_0::*;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

mod instance;
mod physical_device;
mod logical_device;
mod swapchain;
mod pipeline;
mod framebuffer;
mod command_pool;
mod color_object;
mod depth_object;
mod texture;
mod model;
mod buffers;
mod descriptor;
mod command_buffer;
mod sync_object;
mod utility;
mod structs;
mod app;

use app::{AppData, App};
use app::turorial::TutorialApp;

const VALIDATION_ENABLED: bool =
    cfg!(debug_assertions);

const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

// lazy_static! {
//     static ref VERTICES: Vec<Vertex> = vec![
//         Vertex::new(glm::vec3(-0.5, -0.5, 0.0),glm::vec3(1.0, 0.0, 0.0),glm::vec2(1.0, 0.0)),
//         Vertex::new(glm::vec3(0.5, -0.5, 0.0), glm::vec3(0.0, 1.0, 0.0), glm::vec2(0.0, 0.0)),
//         Vertex::new(glm::vec3(0.5, 0.5, 0.0), glm::vec3(0.0, 0.0, 1.0), glm::vec2(0.0, 1.0)),
//         Vertex::new(glm::vec3(-0.5, 0.5, 0.0), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 1.0)),
//         Vertex::new(glm::vec3(-0.5, -0.5, -0.5), glm::vec3(1.0, 0.0, 0.0), glm::vec2(1.0, 0.0)),
//         Vertex::new(glm::vec3(0.5, -0.5, -0.5), glm::vec3(0.0, 1.0, 0.0), glm::vec2(0.0, 0.0)),
//         Vertex::new(glm::vec3(0.5, 0.5, -0.5), glm::vec3(0.0, 0.0, 1.0), glm::vec2(0.0, 1.0)),
//         Vertex::new(glm::vec3(-0.5, 0.5, -0.5), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 1.0)),
//     ];
// }

// const INDICES: &[u16] = &[
//     0, 1, 2, 2, 3, 0,
//     4, 5, 6, 6, 7, 4,
// ];

fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App

    let app_support = TutorialApp::new("resources/viking_room.obj", "resources/viking_room.png");
    let mut app = unsafe { App::create(&window, app_support)? };
    let mut destroying = false;
    let mut minimized = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::MainEventsCleared if !destroying && !minimized  =>
                unsafe { app.render(&window) }.unwrap(),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            }
            // Destroy our Vulkan app.
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                destroying = true;
                *control_flow = ControlFlow::Exit;
                unsafe { app.device.device_wait_idle().unwrap(); }
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
    });
}
