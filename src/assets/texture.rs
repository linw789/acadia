use crate::{buffer::Buffer, image::Image};
use ash::{Device, vk};
use libc::{c_char, c_int, c_uchar, c_void};
use std::{fs, path::PathBuf, ptr::null, vec::Vec};

pub enum TextureSource {
    FilePath(PathBuf),
    Memory((Vec<u8>, vk::Extent3D)),
}

pub struct TextureIngredient {
    pub src: TextureSource,
    pub format: vk::Format,
    pub max_anistropy: f32,
    pub view_component: vk::ComponentMapping,
}

#[derive(Default)]
pub struct Texture {
    pub extent: vk::Extent3D,
    pub image: Image,
    pub sampler: vk::Sampler,
}

impl Texture {
    pub(super) fn destruct(&mut self, device: &Device) {
        self.image.destruct(device);
        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
        self.sampler = vk::Sampler::null();
    }
}

pub(super) fn bake_textures(
    device: &Device,
    cmd_buf: vk::CommandBuffer,
    queue: vk::Queue,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ingredients: &[TextureIngredient],
) -> Vec<Texture> {
    // Create staging buffers for each texture ingredient.
    let (staging_buffers, extents): (Vec<_>, Vec<_>) = ingredients
        .iter()
        .map(|ingredient| {
            let (pixels, extent) = match &ingredient.src {
                TextureSource::FilePath(path) => {
                    let texture_width: c_int = 0;
                    let texture_height: c_int = 0;
                    let original_color_component_count: c_int = 0;
                    let requested_color_component_count = match ingredient.format {
                        vk::Format::R8_UNORM => 1,
                        vk::Format::R8G8_UNORM => 2,
                        vk::Format::R8G8B8_UNORM => 3,
                        vk::Format::R8G8B8A8_SRGB | vk::Format::R8G8B8A8_UNORM => 4,
                        _ => 0,
                    };
                    let image_data = fs::read(path).unwrap();
                    unsafe {
                        let pixels = stbi_load_from_memory(
                            image_data.as_ptr(),
                            image_data.len() as c_int,
                            &texture_width,
                            &texture_height,
                            &original_color_component_count,
                            requested_color_component_count,
                        ) as *const c_uchar;
                        assert!(pixels != null());

                        let byte_size =
                            texture_height * texture_width * requested_color_component_count;
                        let pixels = std::slice::from_raw_parts(pixels, byte_size as usize);

                        let extent = vk::Extent3D {
                            width: texture_width as u32,
                            height: texture_height as u32,
                            depth: 1,
                        };

                        (pixels, extent)
                    }
                }
                TextureSource::Memory((pixels, extent)) => (pixels.as_ref(), *extent),
            };

            let buffer = Buffer::new(
                device,
                pixels.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                memory_properties,
            );
            buffer.copy_data(0, pixels);

            if let TextureSource::FilePath(_) = ingredient.src {
                unsafe {
                    stbi_image_free(pixels.as_ptr() as *mut c_void);
                }
            }

            (buffer, extent)
        })
        .unzip();

    // Create texture holders for each texture ingredient.
    let textures: Vec<_> = ingredients
        .iter()
        .zip(extents.iter())
        .map(|(ingredient, extent)| {
            let image = Image::new_image(
                device,
                memory_properties,
                ingredient.format,
                *extent,
                ingredient.view_component,
            );

            let sampler_createinfo = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .mip_lod_bias(0.0)
                .anisotropy_enable(false)
                .max_anisotropy(ingredient.max_anistropy)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS);

            let sampler = unsafe { device.create_sampler(&sampler_createinfo, None).unwrap() };

            Texture {
                extent: *extent,
                image,
                sampler,
            }
        })
        .collect();

    // Transfer data from staging buffers to images.
    unsafe {
        let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
            .expect("Failed to begin command buffer recording.");

        // Transition the texture image layout to TRANSFER_DST_OPTIMAL
        /*
        let pre_transfer_layout_barriers: Vec<_> = textures
            .iter()
            .map(|texture| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image.image)
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
        device.cmd_pipeline_barrier2(cmd_buf, &pre_transfer_dependencies);
        */

        for (staging_buffer, texture) in staging_buffers.iter().zip(textures.iter()) {
            let layout_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let barriers = [layout_barrier];
            let pre_transfer_dependencies =
                vk::DependencyInfo::default().image_memory_barriers(&barriers);
            device.cmd_pipeline_barrier2(cmd_buf, &pre_transfer_dependencies);

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
                .image_extent(texture.extent);

            device.cmd_copy_buffer_to_image(
                cmd_buf,
                staging_buffer.buf,
                texture.image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );

            let layout_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let barriers = [layout_barrier];
            let post_transfer_dependencies =
                vk::DependencyInfo::default().image_memory_barriers(&barriers);
            device.cmd_pipeline_barrier2(cmd_buf, &post_transfer_dependencies);
        }

        // Transfer the texture image's layout to SHADER_READ_ONLY_OPTIMAL.
        /*
        let post_transfer_layout_barrier: Vec<_> = textures
            .iter()
            .map(|texture| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image.image)
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
        device.cmd_pipeline_barrier2(cmd_buf, &post_transfer_dependencies);
        */

        device.end_command_buffer(cmd_buf).unwrap();

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

    for mut buf in staging_buffers {
        buf.destruct(device);
    }

    textures
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
