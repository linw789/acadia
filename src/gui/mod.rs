pub mod font;

use crate::{buffer::Buffer, common::Vertex2D};
use ash::{Device, util::read_spv, vk};
use font::FontBitmap;
use glam::{Vec2, Vec4, vec2};
use std::{fs::File, vec::Vec};

pub struct Text {
    pub text: String,
    pub start: Vec2,
    pub height: f32,
}

#[derive(Default)]
pub struct DevGui {
    pub font_bitmap: FontBitmap,

    screen_size: Vec2,

    pub vertices: Vec<Vertex2D>,
    pub indices: Vec<u32>,

    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,

    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
}

impl DevGui {
    pub fn new(device: &Device, screen_size: Vec2) -> Self {
        let mut spv_file = File::open("target/shaders/text/text.vert.spv").unwrap();
        let spv = read_spv(&mut spv_file).unwrap();
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&spv);
        let vertex_shader = unsafe {
            device
                .create_shader_module(&vertex_shader_info, None)
                .unwrap()
        };

        let mut spv_file = File::open("target/shaders/text/text.frag.spv").unwrap();
        let spv = read_spv(&mut spv_file).unwrap();
        let fragment_shader_info = vk::ShaderModuleCreateInfo::default().code(&spv);
        let fragment_shader = unsafe {
            device
                .create_shader_module(&fragment_shader_info, None)
                .unwrap()
        };

        Self {
            font_bitmap: FontBitmap::from_truetype("assets/fonts/LiberationSerif-Regular.ttf"),
            screen_size,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_shader,
            fragment_shader,
            ..Default::default()
        }
    }

    pub fn text(&mut self, s: &Text) {
        let mut curr_pos = s.start;
        let scale = 5.0;
        for c in s.text.chars() {
            let glyph_index = (c as u32) - self.font_bitmap.base_codepoint;
            let glyph = &self.font_bitmap.glyphs[glyph_index as usize];
            let glyph_width = (glyph.bottom_right.x - glyph.top_left.x) as f32;
            let glyph_height = (glyph.bottom_right.y - glyph.top_left.y) as f32;
            // let glyph_width_over_height = glyph_width / glyph_height;

            let quad_top_left = curr_pos + (glyph.offset * scale);
            let quad_bottom_left = quad_top_left + vec2(0.0, glyph_height * scale);
            let quad_bottom_right = quad_bottom_left + vec2(glyph_width * scale, 0.0);
            let quad_top_right = quad_bottom_right - vec2(0.0, glyph_height * scale);

            let bitmap_width = self.font_bitmap.width as f32;
            let bitmap_height = self.font_bitmap.height as f32;
            let unit_glyph_width = glyph_width / bitmap_width;
            let unit_glyph_height = glyph_height / bitmap_height;
            let uv_top_left = vec2(
                glyph.top_left.x as f32 / bitmap_width,
                glyph.top_left.y as f32 / bitmap_height,
            );
            let uv_bottom_left = uv_top_left + vec2(0.0, unit_glyph_height);
            let uv_bottom_right = uv_bottom_left + vec2(unit_glyph_width, 0.0);
            let uv_top_right = uv_bottom_right - vec2(0.0, unit_glyph_height);

            curr_pos += vec2(glyph.x_advance * scale, 0.0);

            let start_vertex_index = self.vertices.len() as u32;

            self.vertices.extend(&[
                Vertex2D {
                    pos: (quad_top_left / self.screen_size).into(),
                    color: Vec4::ONE.into(),
                    uv: uv_top_left.into(),
                },
                Vertex2D {
                    pos: (quad_bottom_left / self.screen_size).into(),
                    color: Vec4::ONE.into(),
                    uv: uv_bottom_left.into(),
                },
                Vertex2D {
                    pos: (quad_bottom_right / self.screen_size).into(),
                    color: Vec4::ONE.into(),
                    uv: uv_bottom_right.into(),
                },
                Vertex2D {
                    pos: (quad_top_right / self.screen_size).into(),
                    color: Vec4::ONE.into(),
                    uv: uv_top_right.into(),
                },
            ]);

            self.indices.extend(&[
                // first triangle
                start_vertex_index,
                start_vertex_index + 1,
                start_vertex_index + 3,
                // second triangle
                start_vertex_index + 1,
                start_vertex_index + 2,
                start_vertex_index + 3,
            ]);
        }
    }

    pub fn build_vertex_index_buffer(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) {
        self.vertex_buffer = Buffer::new(
            device,
            (size_of::<Vertex2D>() * self.vertices.len()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &memory_properties,
        );
        self.vertex_buffer.copy_data(0, &self.vertices);

        self.index_buffer = Buffer::new(
            &device,
            (size_of::<u32>() * self.indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &memory_properties,
        );
        self.index_buffer.copy_data(0, &self.indices);
    }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.vertex_shader, None);
            device.destroy_shader_module(self.fragment_shader, None);
        }
        self.vertex_shader = vk::ShaderModule::null();
        self.fragment_shader = vk::ShaderModule::null();

        self.vertex_buffer.destruct(device);
        self.index_buffer.destruct(device);
    }
}
