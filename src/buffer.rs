use crate::util::find_memorytype_index;
use ::ash::{Device, vk};
use std::{
    cell::{RefCell, RefMut},
    ffi::c_void,
    ptr::copy_nonoverlapping,
};

#[derive(Default)]
struct Pointer {
    size: u64,
    ptr: *mut c_void,
}

// Using `RefCell` guarantees that only `Buffer` or `BufferLinearCopy` can mutate its data.
#[derive(Default)]
pub struct Buffer {
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    ptr: RefCell<Pointer>, // mapped pointer
}

/// `BufferLinearCopy` allows sequential copying into `Buffer` objects of different sizes without
/// explicitly managing offset.
pub struct BufferLinearCopy<'a> {
    offset: u64,
    ptr: RefMut<'a, Pointer>,
}

impl Pointer {
    fn copy_value<T: 'static>(&mut self, offset: u64, data: &T) {
        assert!(
            self.size >= (offset + (size_of::<T>() as u64)),
            "buffer size: {}, offset: {}, data size: {}",
            self.size,
            offset,
            size_of::<T>()
        );
        unsafe {
            let dst_ptr = (self.ptr as *mut u8).add(offset as usize) as *mut T;
            copy_nonoverlapping(data as *const T, dst_ptr, 1);
        }
    }

    fn copy_slice<T: 'static>(&mut self, offset: u64, slice: &[T]) {
        assert!(self.size >= (offset + (size_of::<T>() * slice.len()) as u64));
        unsafe {
            let dst_ptr = (self.ptr as *mut u8).add(offset as usize) as *mut T;
            copy_nonoverlapping(slice.as_ptr(), dst_ptr, slice.len());
        }
    }
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
            ptr: RefCell::new(Pointer { size, ptr }),
        }
    }

    pub fn from_slice<T: 'static>(
        device: &Device,
        usage: vk::BufferUsageFlags,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        slice: &[T],
    ) -> Self {
        let mut buf = Self::new(
            device,
            (size_of::<T>() * slice.len()) as u64,
            usage,
            memory_properties,
        );
        buf.copy_slice(0, slice);
        buf
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
    pub fn copy_value<T: 'static>(&mut self, offset: usize, data: &T) {
        let mut ptr = self.ptr.borrow_mut();
        ptr.copy_value(offset as u64, data);
    }

    pub fn copy_slice<T: 'static>(&mut self, offset: usize, slice: &[T]) {
        let mut ptr = self.ptr.borrow_mut();
        ptr.copy_slice(offset as u64, slice);
    }

    pub fn linear_copy(&self, start_offset: u64) -> BufferLinearCopy {
        BufferLinearCopy {
            offset: start_offset,
            ptr: self.ptr.borrow_mut(),
        }
    }
}

impl<'a> BufferLinearCopy<'a> {
    pub fn copy_value<T: 'static>(&mut self, data: &T) {
        self.ptr.copy_value(self.offset, data);
        self.offset += size_of::<T>() as u64;
    }

    pub fn copy_slice<T: 'static>(&mut self, slice: &[T]) {
        self.ptr.copy_slice(self.offset, slice);
        self.offset += (size_of::<T>() * slice.len()) as u64;
    }

    pub fn done(self) {
    }
}
