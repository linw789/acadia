#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

#[derive(Clone, Debug, Default, Copy)]
#[repr(C, packed)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, Default, Copy)]
#[repr(C, packed)]
pub struct Vertex2D {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

/// `'static` constraint makes sure `T` can't be a reference type when `_var` is a temporary value.
/// TODO: this is not ideal. Need to find a universal way to guarantee `T` not a reference type.
pub fn size_of_var<T: 'static>(_var: &T) -> usize {
    size_of::<T>()
}
