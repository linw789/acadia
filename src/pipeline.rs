use crate::{
    common::{Vertex, Vertex2D},
    shader::Program,
};
use ash::{Device, vk};
use std::vec::Vec;

macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

pub fn create_default_graphics_pipeline(
    device: &Device,
    color_attachment_formats: &[vk::Format],
    depth_format: vk::Format,
    program: &Program,
) -> vk::Pipeline {
    let vertex_binding_descs = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];

    let vertex_attrib_descs = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Vertex, pos) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: offset_of!(Vertex, color) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Vertex, normal) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 3,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: offset_of!(Vertex, uv) as u32,
        },
    ];

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding_descs)
        .vertex_attribute_descriptions(&vertex_attrib_descs);

    let mut color_attachment_states = [vk::PipelineColorBlendAttachmentState::default(); 8];
    let color_attachment_count = color_attachment_formats.len();
    for i in 0..color_attachment_count {
        color_attachment_states[i] = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
    }

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&color_attachment_states[0..color_attachment_count]);

    create_graphics_pipeline(
        device,
        &vertex_input_state,
        true,
        color_attachment_formats,
        depth_format,
        &color_blend_state,
        program,
    )
}

pub fn create_dev_gui_graphics_pipeline(
    device: &Device,
    color_attachment_formats: &[vk::Format],
    depth_format: vk::Format,
    program: &Program,
) -> vk::Pipeline {
    let vertex_binding_descs = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(size_of::<Vertex2D>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];

    let vertex_attrib_descs = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: offset_of!(Vertex2D, pos) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: offset_of!(Vertex2D, color) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: offset_of!(Vertex2D, uv) as u32,
        },
    ];

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding_descs)
        .vertex_attribute_descriptions(&vertex_attrib_descs);

    let mut color_attachment_states = [vk::PipelineColorBlendAttachmentState::default(); 8];
    let color_attachment_count = color_attachment_formats.len();
    for i in 0..color_attachment_count {
        color_attachment_states[i] = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
    }

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&color_attachment_states[0..color_attachment_count]);

    create_graphics_pipeline(
        device,
        &vertex_input_state,
        false,
        color_attachment_formats,
        depth_format,
        &color_blend_state,
        program,
    )
}

fn create_graphics_pipeline(
    device: &Device,
    vertex_input_state: &vk::PipelineVertexInputStateCreateInfo,
    depth_enabled: bool,
    color_attachment_formats: &[vk::Format],
    depth_format: vk::Format,
    color_blend_state: &vk::PipelineColorBlendStateCreateInfo,
    program: &Program,
) -> vk::Pipeline {
    let shader_entry_name = c"main";
    let stages: Vec<_> = program
        .shaders
        .iter()
        .map(|shader| {
            vk::PipelineShaderStageCreateInfo::default()
                .module(shader.shader_module)
                .name(shader_entry_name)
                .stage(shader.stage)
        })
        .collect();

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    // Because we use dynamic viewport, we can pass a dummy viewport and scissor to create-info to
    // make Vulkan validation layer happy.
    let viewports = [vk::Viewport::default()];
    let scissor = [vk::Rect2D::default()];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissor);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .cull_mode(vk::CullModeFlags::BACK)
        .line_width(1.0)
        .polygon_mode(vk::PolygonMode::FILL);
    // TODO: what's depth bias?
    // .depth_bias_enable(true);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_enabled)
        .depth_write_enable(depth_enabled)
        .depth_compare_op(vk::CompareOp::GREATER);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let mut rendering_createinfo = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(color_attachment_formats)
        .depth_attachment_format(depth_format);

    let createinfo = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(program.pipeline_layout)
        .push_next(&mut rendering_createinfo);

    let pipelines = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[createinfo], None)
            .expect("Unable to create graphics pipeline")
    };

    pipelines[0]
}
