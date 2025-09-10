use crate::{
    mesh::{Mesh, Bounds},
    texture::{Texture, TextureInfo, TextureSource},
};
use ash::{Device, vk};
use std::{path::PathBuf, vec::Vec};
use glam::Vec3;

#[derive(Default)]
pub struct Entity {
    pub mesh_index: u32,
    pub texture_indices: Vec<u32>,
}

#[derive(Default)]
pub struct Scene {
    pub entities: Vec<Entity>,
    pub meshes: Vec<Mesh>,
    pub textures: Vec<Texture>,
}

pub struct SceneLoader<'a> {
    device: &'a Device,
    device_memory_properties: &'a vk::PhysicalDeviceMemoryProperties,
    max_sampler_anisotropy: f32,
    cmd_buf: vk::CommandBuffer,
    queue: vk::Queue,
}

impl Scene {
    pub fn destruct(&mut self, device: &Device) {
        for texture in self.textures.iter_mut() {
            texture.destruct(device);
        }
        for mesh in self.meshes.iter_mut() {
            mesh.destruct(device);
        }
        self.textures.clear();
        self.meshes.clear();
        self.entities.clear();
    }

    pub fn bounding_box(&self) -> Bounds {
        let mut bounds = Bounds { min: Vec3::MAX, max: Vec3::MIN };

        for mesh in &self.meshes {
            for submesh in &mesh.submeshes {
                bounds.extend(&submesh.bounds);
            }
        }

        bounds
    }
}

impl<'a> SceneLoader<'a> {
    pub fn new(
        device: &'a Device,
        device_memory_properties: &'a vk::PhysicalDeviceMemoryProperties,
        max_sampler_anisotropy: f32,
        cmd_buf: vk::CommandBuffer,
        queue: vk::Queue,
    ) -> Self {
        Self {
            device,
            device_memory_properties,
            max_sampler_anisotropy,
            cmd_buf,
            queue,
        }
    }

    pub fn load_square(self) -> Scene {
        let mut meshes = Vec::new();
        meshes.push(Mesh::from_obj(
            self.device,
            self.device_memory_properties,
            "assets/meshes/square.obj",
        ));

        let textures = {
            let texture_infos = vec![TextureInfo {
                src: TextureSource::FilePath(PathBuf::from("assets/textures/checker.png")),
                format: vk::Format::R8G8B8A8_SRGB,
                max_sampler_anisotropy: self.max_sampler_anisotropy,
                view_component: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
            }];

            Texture::load_textures(
                self.device,
                self.cmd_buf,
                self.queue,
                self.device_memory_properties,
                &texture_infos,
            )
        };

        let entities = vec![Entity {
            mesh_index: 0,
            texture_indices: vec![0],
        }];

        Scene {
            entities,
            meshes,
            textures,
        }
    }

    pub fn load_mario(self) -> Scene {
        let mut meshes = Vec::new();
        meshes.push(Mesh::from_obj(
            self.device,
            self.device_memory_properties,
            "assets/meshes/mario.obj",
        ));

        let textures = {
            let mut texture_infos = Vec::new();
            for i in 0..8 {
                let png_name = format!("assets/textures/mario/mario-{:02}.png", i);
                texture_infos.push(TextureInfo {
                    src: TextureSource::FilePath(PathBuf::from(png_name)),
                    format: vk::Format::R8G8B8A8_SRGB,
                    max_sampler_anisotropy: self.max_sampler_anisotropy,
                    view_component: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    },
                });
            }

            Texture::load_textures(
                self.device,
                self.cmd_buf,
                self.queue,
                self.device_memory_properties,
                &texture_infos,
            )
        };

        let entities = vec![Entity {
            mesh_index: 0,
            texture_indices: (0..textures.len()).map(|x| x as u32).collect(),
        }];

        Scene {
            entities,
            meshes,
            textures,
        }
    }

    pub fn load_shadow_test(self) -> Scene {
        let mut meshes = Vec::new();
        meshes.push(Mesh::from_obj(
            self.device,
            self.device_memory_properties,
            "assets/meshes/shadow-test.obj",
        ));

        let textures = {
            // Create a dummy texture for meshes that don't have textures.
            let dummy_texture_data: [u8; 4] = [186, 193, 196, 255];

            let texture_infos = vec![TextureInfo {
                src: TextureSource::Memory((
                    &dummy_texture_data,
                    vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                )),
                format: vk::Format::R8G8B8A8_SRGB,
                max_sampler_anisotropy: 1.0,
                view_component: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
            }];

            Texture::load_textures(
                self.device,
                self.cmd_buf,
                self.queue,
                self.device_memory_properties,
                &texture_infos,
            )
        };

        let entities = vec![Entity {
            mesh_index: 0,
            texture_indices: vec![0, 0, 0, 0],
        }];

        Scene {
            entities,
            meshes,
            textures,
        }
    }
}
