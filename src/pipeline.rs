use crate::{
    common::{Vertex, Vertex2D},
    shader::Program,
};
use ash::{Device, vk};
use arrayvec::ArrayVec;
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
    set_layouts: Vec<vk::DescriptorSetLayout>,
    pub sets: Vec<vk::DescriptorSet>,
    pub pipeline: vk::Pipeline,
}

impl Pipeline {
    pub fn new_default_graphics_pipeline(
        device: &Device,
        desc_pool: vk::DescriptorPool,
        color_attachment_formats: &[vk::Format],
        depth_format: vk::Format,
        program: &Program,
        max_sampler_count: usize,
    ) -> Pipeline {
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

        let pipeline = Self::new_graphics_pipeline(
            device,
            &vertex_input_state,
            true,
            color_attachment_formats,
            depth_format,
            &color_blend_state,
            program,
        );

        let set_layouts = {
            let per_frame_data_layout = {
                let desc_set_layout_bindings = [vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

                let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&desc_set_layout_bindings);
                unsafe {
                    device
                        .create_descriptor_set_layout(&layout_info, None)
                        .unwrap()
                }
            };

            let sampler_set_layout = {
                let desc_set_layout_bindings = [vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

                let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&desc_set_layout_bindings);
                unsafe {
                    device
                        .create_descriptor_set_layout(&layout_info, None)
                        .unwrap()
                }
            };

            let mut set_layouts = Vec::with_capacity(1 + max_sampler_count);
            set_layouts.push(per_frame_data_layout);
            for _ in 0..max_sampler_count {
                set_layouts.push(sampler_set_layout);
            }

            set_layouts
        };

        let sets = {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(&set_layouts)
                .descriptor_pool(desc_pool);

            unsafe {
                device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap()
            }
        };

        Self {
            set_layouts,
            sets,
            pipeline,
        }
    }

    pub fn new_shadow_graphics_pipeline(
        device: &Device,
        desc_pool: vk::DescriptorPool,
        depth_format: vk::Format,
        program: &Program,
    ) -> Pipeline {
        let vertex_binding_descs = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];

        let vertex_attrib_descs = [vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Vertex, pos) as u32,
        }];

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attrib_descs);

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default().attachments(&[]);

        let pipeline = Self::new_graphics_pipeline(
            device,
            &vertex_input_state,
            true,
            &[],
            depth_format,
            &color_blend_state,
            program,
        );

        let set_layouts = {
            let per_frame_data_layout = {
                let desc_set_layout_bindings = [vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)];

                let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&desc_set_layout_bindings);
                unsafe {
                    device
                        .create_descriptor_set_layout(&layout_info, None)
                        .unwrap()
                }
            };

            vec![per_frame_data_layout]
        };

        let sets = {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(&set_layouts)
                .descriptor_pool(desc_pool);

            unsafe {
                device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap()
            }
        };

        Self {
            set_layouts,
            sets,
            pipeline,
        }
    }

    pub fn new_dev_gui_graphics_pipeline(
        device: &Device,
        desc_pool: vk::DescriptorPool,
        color_attachment_formats: &[vk::Format],
        depth_format: vk::Format,
        program: &Program,
    ) -> Pipeline {
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

        let pipeline = Self::new_graphics_pipeline(
            device,
            &vertex_input_state,
            false,
            color_attachment_formats,
            depth_format,
            &color_blend_state,
            program,
        );

        let set_layouts = {
            let sampler_set_layout = {
                let desc_set_layout_bindings = [vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

                let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&desc_set_layout_bindings);
                unsafe {
                    device
                        .create_descriptor_set_layout(&layout_info, None)
                        .unwrap()
                }
            };

            vec![sampler_set_layout]
        };

        let sets = {
            let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .set_layouts(&set_layouts)
                .descriptor_pool(desc_pool);

            unsafe {
                device
                    .allocate_descriptor_sets(&desc_set_alloc_info)
                    .unwrap()
            }
        };

        Self {
            set_layouts,
            sets,
            pipeline,
        }
    }

    fn new_graphics_pipeline(
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

    pub fn destruct(&mut self, device: &Device) {
        for layout in &self.set_layouts {
            unsafe {
                device.destroy_descriptor_set_layout(*layout, None);
            }
        }
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
    }
}

pub struct PipelineBuilder<'a> {
    device: &'a Device,
    program: &'a Program,
    vertex_input_state: &'a vk::PipelineVertexInputStateCreateInfo<'a>,
    color_attachment_formats: &'a [vk::Format],
    color_blend_state: &'a vk::PipelineColorBlendStateCreateInfo<'a>,
    depth_format: Option<vk::Format>,
    enable_dynamic_depth_bias: bool,
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
            enable_dynamic_depth_bias: false,
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
