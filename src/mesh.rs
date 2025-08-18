use crate::common::Vertex;
use ::tobj::{GPU_LOAD_OPTIONS, load_obj};
use std::{convert::AsRef, fmt::Debug, path::Path, vec::Vec};

#[derive(Default)]
pub struct Mesh {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
}

impl Mesh {
    pub fn from_obj<P: AsRef<Path> + Debug>(obj_file: P) -> Self {
        let (mesh, _material) = load_obj(&obj_file, &GPU_LOAD_OPTIONS)
            .expect(format!("Failed to load obj file at '{:?}'.", obj_file).as_ref());
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

        Self {
            indices: mesh.indices.clone(),
            vertices,
        }
    }
}
