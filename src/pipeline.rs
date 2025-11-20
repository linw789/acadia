use crate::{
    common::{Vertex, Vertex2D},
    shader::Program,
};
use arrayvec::ArrayVec;
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

pub struct PipelineBuilder<'a> {
    device: &'a Device,
    program: &'a Program,
    vertex_input_state: &'a vk::PipelineVertexInputStateCreateInfo<'a>,
    color_attachment_formats: &'a [vk::Format],
    color_blend_state: &'a vk::PipelineColorBlendStateCreateInfo<'a>,
    depth_format: Option<vk::Format>,
    topology: vk::PrimitiveTopology,
    enable_dynamic_depth_bias: bool,
    line_width: f32,
}

impl<'a> PipelineBuilder<'a> {
    pub fn new(
        device: &'a Device,
        program: &'a Program,
        vertex_input_state: &'a vk::PipelineVertexInputStateCreateInfo,
        color_attachment_formats: &'a [vk::Format],
        color_blend_state: &'a vk::PipelineColorBlendStateCreateInfo,
    ) -> Self {
        Self {
            device,
            program,
            vertex_input_state,
            color_attachment_formats,
            color_blend_state,
            depth_format: None,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            enable_dynamic_depth_bias: false,
            line_width: 1.0,
        }
    }

    pub fn depth_format(mut self, format: vk::Format) -> Self {
        self.depth_format = Some(format);
        self
    }

    pub fn enable_dynamic_depth_bias(mut self, enable: bool) -> Self {
        self.enable_dynamic_depth_bias = enable;
        self
    }

    pub fn topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn line_width(mut self, w: f32) -> Self {
        self.line_width = w;
        self
    }

    pub fn build(self) -> vk::Pipeline {
        let shader_entry_name = c"main";
        let stages: Vec<_> = self
            .program
            .shaders
            .iter()
            .map(|shader| {
                vk::PipelineShaderStageCreateInfo::default()
                    .module(shader.shader_module)
                    .name(shader_entry_name)
                    .stage(shader.stage)
            })
            .collect();

        let input_assembly_state =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(self.topology);

        // Because we use dynamic viewport, we can pass a dummy viewport and scissor to create-info to
        // make Vulkan validation layer happy.
        let viewports = [vk::Viewport::default()];
        let scissor = [vk::Rect2D::default()];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissor);

        let line_width = f32::max(1.0, self.line_width);
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .line_width(line_width)
            .polygon_mode(vk::PolygonMode::FILL)
            .depth_bias_enable(self.enable_dynamic_depth_bias);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(self.depth_format.is_some())
            .depth_write_enable(self.depth_format.is_some())
            .depth_compare_op(vk::CompareOp::GREATER);

        let mut dynamic_states = ArrayVec::<_, 3>::new();
        dynamic_states.push(vk::DynamicState::VIEWPORT);
        dynamic_states.push(vk::DynamicState::SCISSOR);
        if self.enable_dynamic_depth_bias {
            dynamic_states.push(vk::DynamicState::DEPTH_BIAS);
        }
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let mut rendering_createinfo = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(self.color_attachment_formats);
        if let Some(depth_format) = self.depth_format {
            rendering_createinfo = rendering_createinfo.depth_attachment_format(depth_format);
        }

        let createinfo = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(self.vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(self.color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(self.program.pipeline_layout)
            .push_next(&mut rendering_createinfo);

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[createinfo], None)
                .expect("Unable to create graphics pipeline")
        };

        pipelines[0]
    }
}
