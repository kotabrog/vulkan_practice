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
    angle: glm::Vec2,
}

impl OneModelApp {
    pub fn new(obj: &str, texture: &str) -> Self {
        OneModelApp {
            obj: obj.to_string(),
            texture: texture.to_string(),
            start: Instant::now(),
            translate_vec: glm::vec3(0.0, 0.0, 0.0),
            angle: glm::vec2(0.0, 0.0),
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
        let mut model = glm::translate(
            &glm::identity(),
            &self.translate_vec,
        );

        model = glm::rotate_z(&model, self.angle.x);
        glm::rotate_y(&model, self.angle.y)
    }

    fn translate(&mut self, vec: glm::Vec3) {
        self.translate_vec += vec;
    }

    fn rotate(&mut self, vec: glm::Vec2) {
        let two_pi = std::f32::consts::PI * 2.0;
        self.angle += vec;
        self.angle.x %= two_pi;
        self.angle.y %= two_pi;
    }
}
