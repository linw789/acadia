use crate::common::Vertex;
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

#[derive(Default)]
pub struct Pipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

#[derive(Default)]
pub struct GraphicsPipelineBuilder<'a> {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    vertex_input_state: Option<vk::PipelineVertexInputStateCreateInfo<'a>>,
    vertex_input_assembly_state: Option<vk::PipelineInputAssemblyStateCreateInfo<'a>>,
    viewport_state: Option<vk::PipelineViewportStateCreateInfo<'a>>,
    rasterization_state: Option<vk::PipelineRasterizationStateCreateInfo<'a>>,
    multisample_state: Option<vk::PipelineMultisampleStateCreateInfo<'a>>,
    depth_stencil_state: Option<vk::PipelineDepthStencilStateCreateInfo<'a>>,
    color_blend_state: Option<vk::PipelineColorBlendStateCreateInfo<'a>>,
    dynamic_state_info: Option<vk::PipelineDynamicStateCreateInfo<'a>>,
    surface_format: Option<vk::Format>,
    rendering_info: Option<vk::PipelineRenderingCreateInfo<'a>>,
    layout: vk::PipelineLayout,
}

impl Pipeline {
    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline(self.pipeline, None);
        }
        self.layout = vk::PipelineLayout::null();
        self.pipeline = vk::Pipeline::null();
    }
}

impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn shader_stages(mut self, stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>) -> Self {
        self.shader_stages = stages;
        self
    }

    pub fn viewport_state(mut self, viewport: vk::PipelineViewportStateCreateInfo<'a>) -> Self {
        self.viewport_state = Some(viewport);
        self
    }

    pub fn layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn surface_format(mut self, format: vk::Format) -> Self {
        self.surface_format = Some(format);
        self
    }

    pub fn build(self, device: &Device) -> Pipeline {
        assert!(self.shader_stages.len() != 0);
        assert!(self.viewport_state.is_some());
        assert!(self.layout != vk::PipelineLayout::null());

        let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];

        let vertex_input_attribute_descriptions = [
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

        let vertex_input_state = if let Some(vert_input) = self.vertex_input_state {
            vert_input
        } else {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_input_binding_descriptions)
                .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
        };

        let vertex_assembly_state = if let Some(vert_assembly) = self.vertex_input_assembly_state {
            vert_assembly
        } else {
            vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            }
        };

        let raster_state = if let Some(raster_state) = self.rasterization_state {
            raster_state
        } else {
            vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                ..Default::default()
            }
        };

        let multisample_state = if let Some(multisample_state) = self.multisample_state {
            multisample_state
        } else {
            vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            }
        };

        let depth_stencil_state = if let Some(depth_stencil) = self.depth_stencil_state {
            depth_stencil
        } else {
            let noop_stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                ..Default::default()
            };

            vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: 1,
                depth_write_enable: 1,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                front: noop_stencil_state,
                back: noop_stencil_state,
                max_depth_bounds: 1.0,
                ..Default::default()
            }
        };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];

        let color_blend_state = if let Some(color_blend) = self.color_blend_state {
            color_blend
        } else {
            vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op(vk::LogicOp::CLEAR)
                .attachments(&color_blend_attachment_states)
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = if let Some(dynamic_state_info) = self.dynamic_state_info {
            dynamic_state_info
        } else {
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states)
        };

        let mut rendering_info = if let Some(rendering_info) = self.rendering_info {
            rendering_info
        } else {
            vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(std::slice::from_ref(
                    self.surface_format.as_ref().unwrap(),
                ))
                .depth_attachment_format(vk::Format::D16_UNORM)
        };

        let pipeline_createinfo = vk::GraphicsPipelineCreateInfo::default()
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&vertex_assembly_state)
            .viewport_state(self.viewport_state.as_ref().unwrap())
            .rasterization_state(&raster_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(self.layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_createinfo], None)
                .expect("Unable to create graphics pipeline")
        };

        Pipeline {
            layout: self.layout,
            pipeline: pipelines[0],
        }
    }
}
