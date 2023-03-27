use vulkanalia::prelude::v1_0::*;

pub mod instance;
pub mod physical_device;
pub mod logical_device;
pub mod swapchain;
pub mod pipeline;
pub mod framebuffer;
pub mod command_pool;
pub mod color_object;
pub mod depth_object;
pub mod texture;
pub mod model;
pub mod buffers;
pub mod descriptor;
pub mod command_buffer;
pub mod sync_object;
pub mod utility;
pub mod structs;
pub mod app;
pub mod event;

const VALIDATION_ENABLED: bool =
    cfg!(debug_assertions);

const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

const MAX_FRAMES_IN_FLIGHT: usize = 2;
