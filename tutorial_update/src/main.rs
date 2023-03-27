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
use winit::event_loop::EventLoop;
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
mod event;

use app::{AppData, App};
use app::turorial::TutorialApp;
use event::EventHandler;

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

    let mut event_handler = EventHandler::new();

    // App

    let app_support = TutorialApp::new("resources/viking_room.obj", "resources/viking_room.png");
    let mut app = unsafe { App::create(&window, app_support)? };
    event_loop.run(move |event, _, control_flow| {
        event_handler.run(&mut app, &window, event, control_flow)
    });
}
