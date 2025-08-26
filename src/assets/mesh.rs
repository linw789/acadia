use crate::{buffer::Buffer, common::Vertex};
use ash::{Device, vk};
use std::{convert::AsRef, path::Path, vec::Vec};
use tobj::{GPU_LOAD_OPTIONS, load_obj};

#[derive(Default)]
pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
}

impl Mesh {
    pub(super) fn from_obj<P: AsRef<Path>>(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        obj_file: P,
    ) -> Self {
        let (mesh, _material) = load_obj(&obj_file.as_ref(), &GPU_LOAD_OPTIONS)
            .expect(format!("Failed to load obj file at '{:?}'.", obj_file.as_ref()).as_ref());
        let mesh = &mesh[0].mesh;

        let vertex_count = mesh.positions.len() / 3;
        let mut vertices = Vec::with_capacity(vertex_count);
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

        let index_buffer = Buffer::new(
            device,
            (size_of::<u32>() * mesh.indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            memory_properties,
        );
        index_buffer.copy_data(0, &mesh.indices);

        let vertex_buffer = Buffer::new(
            device,
            (size_of::<Vertex>() * vertices.len()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            memory_properties,
        );
        vertex_buffer.copy_data(0, &vertices);

        Self {
            vertex_buffer,
            index_buffer,
        }
    }

    pub(super) fn destruct(&mut self, device: &Device) {
        self.index_buffer.destruct(device);
        self.vertex_buffer.destruct(device);
    }
}
