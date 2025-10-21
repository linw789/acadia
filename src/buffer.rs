use crate::util::find_memorytype_index;
use ::ash::{Device, vk};
use std::{ffi::c_void, ptr::copy_nonoverlapping};

#[derive(Default)]
pub struct Buffer {
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: u64,
    pub ptr: *mut c_void, // mapped pointer
}

impl Buffer {
    pub fn new(
        device: &Device,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        let buf_createinfo = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buf = unsafe { device.create_buffer(&buf_createinfo, None).unwrap() };

        let vert_buf_mem_req = unsafe { device.get_buffer_memory_requirements(buf) };
        let vert_buf_mem_index = find_memorytype_index(
            &vert_buf_mem_req,
            memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Failed to find suitable memorytype for the vertex buffer.");

        let required_size = vert_buf_mem_req.size;
        let vert_mem_alloc_info = vk::MemoryAllocateInfo {
            allocation_size: required_size,
            memory_type_index: vert_buf_mem_index,
            ..Default::default()
        };
        let mem = unsafe { device.allocate_memory(&vert_mem_alloc_info, None).unwrap() };

        unsafe {
            device.bind_buffer_memory(buf, mem, 0).unwrap();
        }

        let ptr = unsafe {
            device
                .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())
                .unwrap()
            // No need to unmap memory after copy (persistent mapping).
        };

        Self {
            buf,
            mem,
            size,
            ptr,
        }
    }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.free_memory(self.mem, None);
            device.destroy_buffer(self.buf, None);
        }
        self.mem = vk::DeviceMemory::null();
        self.buf = vk::Buffer::null();
    }

    /// The `'static` constraint prevents `data` from being a temporary value of a reference type,
    /// but not if `data` is a static reference. `data` should never be a reference, so this is not
    /// perfect.
    pub fn copy_value<T: 'static>(&self, offset: usize, data: &T) {
        assert!(
            self.size >= (offset + size_of::<T>()) as u64,
            "buffer size: {}, offset: {}, data size: {}",
            self.size,
            offset,
            size_of::<T>()
        );
        unsafe {
            let dst_ptr = (self.ptr as *mut u8).add(offset) as *mut T;
            copy_nonoverlapping(data as *const T, dst_ptr, 1);
        }
    }

    pub fn copy_slice<T: 'static>(&self, offset: usize, slice: &[T]) {
        assert!(self.size >= (offset + size_of::<T>() * slice.len()) as u64);
        unsafe {
            let dst_ptr = (self.ptr as *mut u8).add(offset) as *mut T;
            copy_nonoverlapping(slice.as_ptr(), dst_ptr, slice.len());
        }
    }
}
