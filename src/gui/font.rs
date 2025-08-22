use glam::{U16Vec2, Vec2, u16vec2, vec2};
use libc::{c_float, c_int, c_uchar, c_ushort};
use std::{fs, path::Path, vec::Vec};

#[derive(Default)]
pub struct Glyph {
    // The pixel position of the top-left corner of the glyph in the font bitmap.
    pub top_left: U16Vec2,
    // The pixel position of the bottom-right corner of the glyph in the font bitmap.
    pub bottom_right: U16Vec2,
    // The offset that should be applied to the position of the glyph quad rendered in screen pixel
    // space.
    pub offset: Vec2,
    // The offset that should be applied to the x position of the next glyph quad rendered in
    // screen pixel space.
    pub x_advance: f32,
}

#[derive(Default)]
pub struct FontBitmap {
    pub width: u32,
    pub height: u32,
    pub base_codepoint: u32,
    pub pixels: Vec<u8>,
    pub glyphs: Vec<Glyph>,
}

impl FontBitmap {
    pub fn from_truetype<P: AsRef<Path>>(ttf_path: P) -> Self {
        let ttf_data = fs::read(ttf_path).unwrap();
        let ttf_data_offset: c_int = 0;
        let pixel_height: c_float = 30.0;
        let bitmap_width: c_int = 300;
        let bitmap_height: c_int = 130;
        let bitmap: Vec<u8> = vec![0; (bitmap_width * bitmap_height) as usize];
        let first_char: c_int = 32;
        let num_chars: c_int = 96;
        let char_data: Vec<STBTT_BackedChar> =
            vec![STBTT_BackedChar::default(); num_chars as usize];

        unsafe {
            let result = stbtt_BakeFontBitmap(
                ttf_data.as_ptr(),
                ttf_data_offset,
                pixel_height,
                bitmap.as_ptr(),
                bitmap_width,
                bitmap_height,
                first_char,
                num_chars,
                char_data.as_ptr(),
            );
            assert!(result > 0);
        }

        let glyphs = char_data
            .into_iter()
            .map(|c| Glyph {
                top_left: u16vec2(c.x0, c.y0),
                bottom_right: u16vec2(c.x1, c.y1),
                offset: vec2(c.xoff, c.yoff),
                x_advance: c.xadvance,
            })
            .collect();

        Self {
            width: bitmap_width as u32,
            height: bitmap_height as u32,
            base_codepoint: first_char as u32,
            pixels: bitmap,
            glyphs,
        }
    }
}

#[derive(Default, Clone)]
#[allow(non_camel_case_types, dead_code)]
#[repr(C)]
struct STBTT_BackedChar {
    x0: c_ushort,
    y0: c_ushort,
    x1: c_ushort,
    y1: c_ushort,
    xoff: c_float,
    yoff: c_float,
    xadvance: c_float,
}

#[link(name = "stb_truetype")]
unsafe extern "C" {
    fn stbtt_BakeFontBitmap(
        data: *const c_uchar,
        offset: c_int,
        pixel_height: c_float,
        pixels: *const c_uchar,
        pw: c_int,
        ph: c_int,
        first_char: c_int,
        num_chars: c_int,
        chardata: *const STBTT_BackedChar,
    ) -> c_int;
}
