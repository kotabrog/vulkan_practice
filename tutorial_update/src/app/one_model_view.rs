use std::time::Instant;

use anyhow::Result;
use nalgebra_glm as glm;
use vulkanalia::prelude::v1_0::*;

use super::{AppData, AppSupport};
use crate::model::load_model;
use crate::texture::create_texture_image;

#[derive(Clone, Debug)]
pub struct OneModelApp {
    obj: String,
    texture: String,
    start: Instant,
    translate_vec: glm::Vec3,
}

impl OneModelApp {
    pub fn new(obj: &str, texture: &str) -> Self {
        OneModelApp {
            obj: obj.to_string(),
            texture: texture.to_string(),
            start: Instant::now(),
            translate_vec: glm::vec3(0.0, 0.0, 0.0),
        }
    }
}

impl AppSupport for OneModelApp {
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

    fn make_opacity(&self, _model_index: usize) -> f32 {
        1.0
    }

    fn make_model_matrix(&self, _model_index: usize) -> glm::Mat4 {
        let model = glm::translate(
            &glm::identity(),
            &self.translate_vec,
        );

        let time = self.get_elapsed_time();

        glm::rotate(
            &model,
            time * glm::radians(&glm::vec1(90.0))[0],
            &glm::vec3(0.0, 0.0, 1.0),
        )
    }

    fn translate(&mut self, vec: glm::Vec3) {
        self.translate_vec += vec;
    }
}
