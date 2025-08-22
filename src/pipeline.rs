use crate::common::{Vertex, Vertex2D};
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

#[derive(Default, Clone, Copy)]
pub struct Pipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

#[derive(Default)]
pub struct GraphicsPipelineInfo<'a> {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,

    vert_input_binding_descs: Vec<vk::VertexInputBindingDescription>,
    vert_input_attrib_descs: Vec<vk::VertexInputAttributeDescription>,
    vert_input_state: Option<vk::PipelineVertexInputStateCreateInfo<'a>>,

    vert_input_assembly_state: Option<vk::PipelineInputAssemblyStateCreateInfo<'a>>,
    viewport_state: Option<vk::PipelineViewportStateCreateInfo<'a>>,
    rasterization_state: Option<vk::PipelineRasterizationStateCreateInfo<'a>>,
    multisample_state: Option<vk::PipelineMultisampleStateCreateInfo<'a>>,
    depth_stencil_state: Option<vk::PipelineDepthStencilStateCreateInfo<'a>>,

    color_blend_attachment_states: Vec<vk::PipelineColorBlendAttachmentState>,
    color_blend_state: Option<vk::PipelineColorBlendStateCreateInfo<'a>>,

    dynamic_states: Vec<vk::DynamicState>,
    dynamic_state_info: Option<vk::PipelineDynamicStateCreateInfo<'a>>,

    surface_format: Option<vk::Format>,

    pub layout: vk::PipelineLayout,
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

    pub fn create_graphics_pipelines(
        device: &Device,
        pipeline_infos: &[GraphicsPipelineInfo],
    ) -> Vec<Pipeline> {
        let mut vert_input_states = Vec::with_capacity(pipeline_infos.len());
        let mut color_blend_states = Vec::with_capacity(pipeline_infos.len());
        let mut dynamic_state_infos = Vec::with_capacity(pipeline_infos.len());
        let mut rendering_infos = Vec::with_capacity(pipeline_infos.len());
        for info in pipeline_infos.iter() {
            vert_input_states.push(
                vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(&info.vert_input_binding_descs)
                    .vertex_attribute_descriptions(&info.vert_input_attrib_descs),
            );

            color_blend_states.push(
                vk::PipelineColorBlendStateCreateInfo::default()
                    .logic_op(vk::LogicOp::CLEAR)
                    .attachments(&info.color_blend_attachment_states),
            );

            dynamic_state_infos.push(
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&info.dynamic_states),
            );

            rendering_infos.push(
                vk::PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(std::slice::from_ref(
                        info.surface_format.as_ref().unwrap(),
                    ))
                    .depth_attachment_format(vk::Format::D16_UNORM),
            );
        }

        let mut vk_pipeline_infos = Vec::with_capacity(pipeline_infos.len());
        for ((i, info), renderinginfo) in pipeline_infos
            .iter()
            .enumerate()
            .zip(rendering_infos.iter_mut())
        {
            vk_pipeline_infos.push(
                vk::GraphicsPipelineCreateInfo::default()
                    .stages(&info.shader_stages)
                    .vertex_input_state(&vert_input_states[i])
                    .input_assembly_state(info.vert_input_assembly_state.as_ref().unwrap())
                    .viewport_state(info.viewport_state.as_ref().unwrap())
                    .rasterization_state(info.rasterization_state.as_ref().unwrap())
                    .multisample_state(info.multisample_state.as_ref().unwrap())
                    .depth_stencil_state(info.depth_stencil_state.as_ref().unwrap())
                    .color_blend_state(&color_blend_states[i])
                    .dynamic_state(&dynamic_state_infos[i])
                    .layout(info.layout)
                    .push_next(renderinginfo),
            );
        }

        let vk_pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &vk_pipeline_infos, None)
                .expect("Unable to create graphics pipeline")
        };

        let mut pipelines = Vec::new();
        for (i, p) in vk_pipelines.iter().enumerate() {
            pipelines.push(Pipeline {
                layout: vk_pipeline_infos[i].layout,
                pipeline: *p,
            });
        }

        pipelines
    }
}

impl<'a> GraphicsPipelineInfo<'a> {
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

    pub fn build(mut self) -> Self {
        assert!(self.shader_stages.len() != 0);
        assert!(self.viewport_state.is_some());
        assert!(self.layout != vk::PipelineLayout::null());

        if self.vert_input_state.is_none() {
            self.vert_input_binding_descs = vec![vk::VertexInputBindingDescription {
                binding: 0,
                stride: size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }];

            self.vert_input_attrib_descs = vec![
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
        }

        if self.vert_input_assembly_state.is_none() {
            self.vert_input_assembly_state = Some(vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            });
        }

        if self.rasterization_state.is_none() {
            self.rasterization_state = Some(vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                ..Default::default()
            });
        }

        if self.multisample_state.is_none() {
            self.multisample_state = Some(vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            });
        }

        if self.depth_stencil_state.is_none() {
            let noop_stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                ..Default::default()
            };

            self.depth_stencil_state = Some(vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: 1,
                depth_write_enable: 1,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                front: noop_stencil_state,
                back: noop_stencil_state,
                max_depth_bounds: 1.0,
                ..Default::default()
            });
        }

        if self.color_blend_state.is_none() {
            self.color_blend_attachment_states = vec![vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }];
        }

        if self.dynamic_state_info.is_none() {
            self.dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        }

        self
    }

    pub fn build_dev_gui(mut self) -> Self {
        assert!(self.shader_stages.len() != 0);
        assert!(self.viewport_state.is_some());
        assert!(self.layout != vk::PipelineLayout::null());

        self.vert_input_binding_descs = vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vertex2D>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];

        self.vert_input_attrib_descs = vec![
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

        self.vert_input_assembly_state = Some(
            vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
        );

        if self.rasterization_state.is_none() {
            self.rasterization_state = Some(vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                ..Default::default()
            });
        }

        if self.multisample_state.is_none() {
            self.multisample_state = Some(vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            });
        }

        if self.depth_stencil_state.is_none() {
            let noop_stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                ..Default::default()
            };

            self.depth_stencil_state = Some(vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: 0,
                front: noop_stencil_state,
                back: noop_stencil_state,
                max_depth_bounds: 1.0,
                ..Default::default()
            });
        }

        if self.color_blend_state.is_none() {
            self.color_blend_attachment_states = vec![vk::PipelineColorBlendAttachmentState {
                blend_enable: 1,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }];
        }

        if self.dynamic_state_info.is_none() {
            self.dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        }

        self
    }
}
