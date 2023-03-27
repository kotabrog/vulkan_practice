use std::time::Instant;

use anyhow::Result;
use nalgebra_glm as glm;
use vulkanalia::prelude::v1_0::*;

use super::{AppData, AppSupport};
use crate::model::load_model;
use crate::texture::create_texture_image;

#[derive(Clone, Debug)]
pub struct TutorialApp {
    obj: String,
    texture: String,
    start: Instant,
}

impl TutorialApp {
    pub fn new(obj: &str, texture: &str) -> Self {
        TutorialApp {
            obj: obj.to_string(),
            texture: texture.to_string(),
            start: Instant::now(),
        }
    }
}

impl AppSupport for TutorialApp {
    fn make_model(&self, data: &mut AppData) -> Result<()> {
        load_model(data, &self.obj)
    }

    unsafe fn create_texture_image(
        &self,
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        create_texture_image(instance, device, data, &self.texture)
    }

    fn get_elapsed_time(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    fn make_view_obj(&self) -> glm::Mat4 {
        glm::look_at(
            &glm::vec3(6.0, 0.0, 2.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 0.0, 1.0),
        )
    }

    fn make_opacity(&self, model_index: usize) -> f32 {
        (model_index + 1) as f32 * 0.25
    }

    fn make_model_matrix(&self, model_index: usize) -> glm::Mat4 {
        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        let model = glm::translate(
            &glm::identity(),
            &glm::vec3(0.0, y, z),
        );

        let time = self.get_elapsed_time();

        glm::rotate(
            &model,
            time * glm::radians(&glm::vec1(90.0))[0],
            &glm::vec3(0.0, 0.0, 1.0),
        )
    }
}
