use crate::{buffer::Buffer, common::Vertex};
use ash::{Device, vk};
use std::{convert::AsRef, path::Path, vec::Vec};
use tobj::{GPU_LOAD_OPTIONS, Model, load_obj};

#[derive(Default)]
pub struct SubMesh {
    pub index_count: u32,
    pub index_offset: u32,
    pub vertex_offset: i32,
}

#[derive(Default)]
pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub submeshes: Vec<SubMesh>,
}

impl Mesh {
    pub(super) fn from_obj<P: AsRef<Path>>(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        obj_file: P,
    ) -> Self {
        let (models, _material) = load_obj(&obj_file.as_ref(), &GPU_LOAD_OPTIONS)
            .expect(format!("Failed to load obj file at '{:?}'.", obj_file.as_ref()).as_ref());

        let mut submeshes = Vec::with_capacity(models.len());

        let total_vertex_count = models
            .iter()
            .map(|model| model.mesh.positions.len() / 3)
            .sum();

        let total_index_count = models.iter().map(|model| model.mesh.indices.len()).sum();

        let mut vertices = Vec::with_capacity(total_vertex_count);
        let mut indices = Vec::with_capacity(total_index_count);

        // Value added to each vertex index by GPU before indexing into vertex buffer.
        let mut vertex_offset = 0;
        let mut index_offset = 0;
        for Model { mesh, name } in models {
            let vertex_count = mesh.positions.len() / 3;
            let index_count = mesh.indices.len();

            for vi in 0..vertex_count {
                let pos = [
                    mesh.positions[vi * 3 + 0],
                    mesh.positions[vi * 3 + 1],
                    mesh.positions[vi * 3 + 2],
                ];
                let color = if mesh.vertex_color.len() > 0 {
                    [
                        mesh.vertex_color[vi * 3 + 0],
                        mesh.vertex_color[vi * 3 + 1],
                        mesh.vertex_color[vi * 3 + 2],
                        1.0,
                    ]
                } else {
                    [186.0 / 255.0, 193.0 / 255.0, 196.0 / 255.0, 1.0]
                };
                let normal = if mesh.normals.len() > 0 {
                    [
                        mesh.normals[vi * 3 + 0],
                        mesh.normals[vi * 3 + 1],
                        mesh.normals[vi * 3 + 2],
                    ]
                } else {
                    [0.0; 3]
                };
                let uv = if mesh.texcoords.len() > 0 {
                    [mesh.texcoords[vi * 2 + 0], mesh.texcoords[vi * 2 + 1]]
                } else {
                    [0.0; 2]
                };
                vertices.push(Vertex {
                    pos,
                    color,
                    normal,
                    uv,
                });
            }

            indices.extend(mesh.indices);

            submeshes.push(SubMesh {
                index_count: index_count as u32,
                index_offset,
                vertex_offset,
            });

            vertex_offset += vertex_count as i32;
            index_offset += index_count as u32;
        }

        let vertex_buffer = Buffer::new(
            device,
            (size_of::<Vertex>() * vertices.len()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            memory_properties,
        );
        vertex_buffer.copy_data(0, &vertices);

        let index_buffer = Buffer::new(
            device,
            (size_of::<u32>() * total_index_count) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            memory_properties,
        );
        index_buffer.copy_data(0, &indices);

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            submeshes,
        }
    }

    pub(super) fn destruct(&mut self, device: &Device) {
        self.index_buffer.destruct(device);
        self.vertex_buffer.destruct(device);
    }
}
