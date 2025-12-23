use crate::{buffer::Buffer, util::find_memorytype_index};
use ::ash::{Device, vk};
use libc::{c_char, c_int, c_uchar, c_void};
use std::{fs, path::Path, ptr::null, sync::Arc};

pub struct ImagePool {
    device: Arc<Device>,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    images: Vec<Option<Image>>,
}

#[derive(Default)]
pub struct Image {
    pub extent: vk::Extent3D,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
}

pub struct ImageCreateParam {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub usage: vk::ImageUsageFlags,
}

impl ImagePool {
    pub fn new(
        device: Arc<Device>,
        device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        Self {
            device,
            device_memory_properties,
            images: Vec::new(),
        }
    }

    pub fn new_depth_image(&mut self, extent: vk::Extent2D, format: vk::Format) -> u32 {
        let (image, view, memory) = unsafe {
            let depth_image_createinfo = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let depth_image = self
                .device
                .create_image(&depth_image_createinfo, None)
                .unwrap();
            let depth_image_memory_req = self.device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &self.device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index);

            let depth_image_memory = self
                .device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            self.device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .expect("Unable to bind depth image memory.");

            let depth_image_view_info = vk::ImageViewCreateInfo::default()
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                )
                .image(depth_image)
                .format(depth_image_createinfo.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            let depth_image_view = self
                .device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            (depth_image, depth_image_view, depth_image_memory)
        };

        self.add_image(Image {
            extent: extent.into(),
            image,
            view,
            memory,
        })
    }

    pub fn new_image(
        &mut self,
        createparam: &ImageCreateParam,
    ) -> u32 {
        self.add_image(Image::new(
            &self.device,
            &self.device_memory_properties,
            createparam,
        ))
    }

    pub fn new_images_from_bytes(
        &mut self,
        byte_slices: &[(&[u8], vk::Extent3D)],
        format: vk::Format,
        view_components: vk::ComponentMapping,
        cmd_buf: vk::CommandBuffer,
        queue: vk::Queue,
    ) -> Vec<u32> {
        let mut staging_buffers = Vec::with_capacity(byte_slices.len());
        for (bytes, _ext) in byte_slices {
            let buf = Buffer::from_slice(
                &self.device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                &self.device_memory_properties,
                bytes,
            );
            staging_buffers.push(buf);
        }

        let mut images: Vec<_> = byte_slices
            .iter()
            .map(|(_bytes, ext)| {
                let createparam = ImageCreateParam {
                    extent: *ext,
                    format,
                    components: view_components,
                    ..Default::default()
                };
                Image::new(
                    &self.device,
                    &self.device_memory_properties,
                    &createparam,
                )
            })
            .collect();

        // Transfer data from staging buffers to images.
        unsafe {
            let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
                .expect("Failed to begin command buffer recording.");

            // Transition the image layout to TRANSFER_DST_OPTIMAL
            let pre_transfer_layout_barriers: Vec<_> = images
                .iter()
                .map(|image| {
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image.image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                })
                .collect();
            let pre_transfer_dependencies =
                vk::DependencyInfo::default().image_memory_barriers(&pre_transfer_layout_barriers);
            self.device
                .cmd_pipeline_barrier2(cmd_buf, &pre_transfer_dependencies);

            for (staging_buffer, image) in staging_buffers.iter().zip(images.iter()) {
                // Copy the texture from buffer to image.
                let copy_region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(image.extent);

                self.device.cmd_copy_buffer_to_image(
                    cmd_buf,
                    staging_buffer.buf,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    std::slice::from_ref(&copy_region),
                );
            }

            // Transfer the image's layout to SHADER_READ_ONLY_OPTIMAL.
            let post_transfer_layout_barrier: Vec<_> = images
                .iter()
                .map(|image| {
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image.image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                })
                .collect();
            let post_transfer_dependencies =
                vk::DependencyInfo::default().image_memory_barriers(&post_transfer_layout_barrier);
            self.device
                .cmd_pipeline_barrier2(cmd_buf, &post_transfer_dependencies);

            self.device.end_command_buffer(cmd_buf).unwrap();

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&[])
                .wait_dst_stage_mask(&[])
                .command_buffers(std::slice::from_ref(&cmd_buf))
                .signal_semaphores(&[]);

            self.device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .expect("Failed to queue submit for transferring texture image.");

            self.device.queue_wait_idle(queue).unwrap();
        }

        for mut buf in staging_buffers {
            buf.destruct(&self.device);
        }

        let image_indices: Vec<_> = images
            .drain(0..)
            .map(|image| self.add_image(image))
            .collect();
        image_indices
    }

    pub fn new_images_from_files<P: AsRef<Path>>(
        &mut self,
        file_paths: &[P],
        format: vk::Format,
        view_components: vk::ComponentMapping,
        cmd_buf: vk::CommandBuffer,
        queue: vk::Queue,
    ) -> Vec<u32> {
        let byte_slices: Vec<_> = file_paths
            .iter()
            .map(|path| {
                let width: c_int = 0;
                let height: c_int = 0;
                let original_color_component_count: c_int = 0;
                let requested_color_component_count = match format {
                    vk::Format::R8_UNORM => 1,
                    vk::Format::R8G8_UNORM => 2,
                    vk::Format::R8G8B8_UNORM => 3,
                    vk::Format::R8G8B8A8_SRGB | vk::Format::R8G8B8A8_UNORM => 4,
                    _ => 0,
                };
                let image_data = fs::read(path).unwrap();
                let (pixels, extent) = unsafe {
                    let pixels = stbi_load_from_memory(
                        image_data.as_ptr(),
                        image_data.len() as c_int,
                        &width,
                        &height,
                        &original_color_component_count,
                        requested_color_component_count,
                    ) as *const c_uchar;
                    assert!(pixels != null());

                    let byte_size = height * width * requested_color_component_count;
                    let pixels = std::slice::from_raw_parts(pixels, byte_size as usize);

                    let extent = vk::Extent3D {
                        width: width as u32,
                        height: height as u32,
                        depth: 1,
                    };

                    (pixels, extent)
                };

                (pixels, extent)
            })
            .collect();

        let image_indices =
            self.new_images_from_bytes(&byte_slices, format, view_components, cmd_buf, queue);

        for (bytes, _ext) in byte_slices {
            unsafe {
                stbi_image_free(bytes.as_ptr() as *mut c_void);
            }
        }

        image_indices
    }

    pub fn get_at_index(&self, index: u32) -> &Image {
        &self.images[index as usize].as_ref().unwrap()
    }

    pub fn delete_at_index(&mut self, index: u32) {
        if let Some(i) = self.images[index as usize].as_mut() {
            i.destruct(&self.device);
        }
    }

    pub fn destruct(&mut self) {
        for image in self.images.iter_mut() {
            if let Some(i) = image {
                i.destruct(&self.device);
            }
        }
        self.images.clear();
    }

    fn add_image(&mut self, image: Image) -> u32 {
        let index = if let Some((index, slot)) = self
            .images
            .iter_mut()
            .enumerate()
            .find(|(_, i)| i.is_none())
        {
            *slot = Some(image);
            index
        } else {
            self.images.push(Some(image));
            self.images.len() - 1
        };
        index as u32
    }
}

impl Image {
    fn new(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        createparam: &ImageCreateParam,
    ) -> Self {
        let image_createinfo = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(createparam.format)
            .extent(createparam.extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(createparam.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&image_createinfo, None).unwrap() };

        let image_memory_req = unsafe { device.get_image_memory_requirements(image) };
        let image_memory_index = find_memorytype_index(
            &image_memory_req,
            memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("Unable to find suitable memory index for depth image.");

        let image_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(image_memory_req.size)
            .memory_type_index(image_memory_index);

        let memory = unsafe { device.allocate_memory(&image_allocate_info, None).unwrap() };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("Unable to bind depth image memory.");
        }

        let view_createinfo = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(createparam.format)
            .components(createparam.components)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { device.create_image_view(&view_createinfo, None).unwrap() };

        Self {
            extent: createparam.extent,
            image,
            view,
            memory,
        }
    }

    fn destruct(&mut self, device: &Device) {
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }

        self.extent = vk::Extent3D::default();
        self.image = vk::Image::null();
        self.view = vk::ImageView::null();
        self.memory = vk::DeviceMemory::null();
    }
}

impl Default for ImageCreateParam {
    fn default() -> Self {
        Self {
            extent: vk::Extent3D { width: 0, height: 0, depth: 0 },
            format: vk::Format::R32G32B32A32_SFLOAT,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        }
    }
}

#[link(name = "stb_image")]
unsafe extern "C" {
    fn stbi_load_from_memory(
        data: *const c_uchar,
        len: c_int,
        w: *const c_int,
        h: *const c_int,
        comp_n: *const c_int,
        desire_comp_n: c_int,
    ) -> *const c_char;

    fn stbi_image_free(data: *mut c_void);
}
