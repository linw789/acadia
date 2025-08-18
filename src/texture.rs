use crate::{buffer::Buffer, image::Image};
use ash::{Device, vk};
use libc::{c_char, c_int, c_void};
use std::{ffi::CString, os::unix::ffi::OsStrExt, path::Path, ptr::null};

#[derive(Default)]
pub struct Texture {
    pub image: Image,
    pub sampler: vk::Sampler,
}

impl Texture {
    pub fn new<P: AsRef<Path>>(
        device: &Device,
        cmd_buf: vk::CommandBuffer,
        queue: vk::Queue,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        max_sampler_anisotropy: f32,
        texture_path: P,
    ) -> Self {
        let texture_width: c_int = 0;
        let texture_height: c_int = 0;
        let _color_component_count: c_int = 0;
        let pixels = unsafe {
            let path = CString::new(texture_path.as_ref().as_os_str().as_bytes()).unwrap();
            stbi_load(
                path.as_ptr(),
                &texture_width,
                &texture_height,
                &_color_component_count,
                STBI_ColorFormat::RgbAlpha as i32,
            )
        };
        assert!(pixels != null());

        let texture_size = texture_width * texture_height * 4;
        let mut staging_buffer = Buffer::new(
            device,
            texture_size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            memory_properties,
        );
        staging_buffer.copy_data(0, unsafe {
            std::slice::from_raw_parts(pixels, texture_size as usize)
        });

        unsafe {
            stbi_image_free(pixels as *mut c_void);
        }

        let image = Image::new_texture_image(
            device,
            memory_properties,
            vk::Extent2D {
                width: texture_width as u32,
                height: texture_height as u32,
            },
        );

        // Upload GPU commands to transfer texture.

        unsafe {
            device
                .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .unwrap();

            let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
                .expect("Failed to begin command buffer recording.");
        }

        // Transition the texture image layout to TRANSFER_DST_OPTIMAL
        unsafe {
            let barrier = vk::ImageMemoryBarrier2::default()
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
                });
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            device.cmd_pipeline_barrier2(cmd_buf, &dependency_info);
        }

        // Copy the texture from buffer to image.
        unsafe {
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
                .image_extent(vk::Extent3D {
                    width: texture_width as u32,
                    height: texture_height as u32,
                    depth: 1,
                });

            device.cmd_copy_buffer_to_image(
                cmd_buf,
                staging_buffer.buf,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );
        }

        // Transfer the texture image's layout to SHADER_READ_ONLY_OPTIMAL.
        unsafe {
            let barrier = vk::ImageMemoryBarrier2::default()
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
                });
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            device.cmd_pipeline_barrier2(cmd_buf, &dependency_info);
        }

        unsafe {
            device.end_command_buffer(cmd_buf).unwrap();
        }

        unsafe {
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&[])
                .wait_dst_stage_mask(&[])
                .command_buffers(std::slice::from_ref(&cmd_buf))
                .signal_semaphores(&[]);

            device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .expect("Failed to queue submit for transferring texture image.");

            device.queue_wait_idle(queue).unwrap();
        }

        staging_buffer.destruct(device);

        let sampler_createinfo = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .max_anisotropy(max_sampler_anisotropy)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS);

        let sampler = unsafe { device.create_sampler(&sampler_createinfo, None).unwrap() };

        Self { image, sampler }
    }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
        self.sampler = vk::Sampler::null();

        self.image.destruct(device);
    }
}

/// This enum must be exactly the same as the STBI_ enum defined in stb_image.h.
#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
enum STBI_ColorFormat {
    Default = 0,
    Grey = 1,
    GreyAlpha = 2,
    Rgb = 3,
    RgbAlpha = 4,
}

#[link(name = "stb_image")]
unsafe extern "C" {
    fn stbi_load(
        filename: *const c_char,
        w: *const c_int,
        h: *const c_int,
        comp_n: *const c_int,
        desire_comp_n: c_int,
    ) -> *const c_char;

    fn stbi_image_free(data: *mut c_void);
}
