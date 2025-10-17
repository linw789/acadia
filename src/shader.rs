use ash::{Device, util::read_spv, vk};
use spirv::{Decoration, ExecutionModel, Op, StorageClass};
use std::{
    collections::HashMap,
    fs::{File, read_dir},
    path::Path,
    rc::Rc,
    vec::Vec,
};

pub struct Shader {
    pub stage: vk::ShaderStageFlags,
    pub shader_module: vk::ShaderModule,
    descriptor_infos: Vec<DescriptorInfo>,
    use_descriptor_array: bool,
    use_push_constants: bool,
}

#[derive(Default)]
pub struct Program {
    pub shaders: Vec<Rc<Shader>>,
    pub bind_point: vk::PipelineBindPoint,
    desc_set_layouts: [vk::DescriptorSetLayout; 4],
    desc_set_layout_count: usize,
    pub pipeline_layout: vk::PipelineLayout,
}

struct DescriptorInfo {
    set: u32,
    binding: u32,
    kind: vk::DescriptorType,
}

#[derive(Clone)]
struct SpvId {
    opcode: Op,
    type_id: u32,
    storage_class: StorageClass,
    binding: u32,
    set: u32,
    constant: u32,
}

pub fn load_shaders<P: AsRef<Path>>(
    device: &Device,
    shaders_dir: P,
    root_dir: P,
    shader_set: &mut HashMap<String, Rc<Shader>>,
) {
    assert!(shaders_dir.as_ref().is_dir());
    for entry in read_dir(shaders_dir).unwrap() {
        let entry = entry.unwrap();
        let shader_path = entry.path();
        if shader_path.is_dir() {
            load_shaders(device, shader_path.as_path(), root_dir.as_ref(), shader_set);
        } else {
            let shader = Shader::new(device, &shader_path);
            let mut shader_path = shader_path;
            shader_path.set_extension("");
            let mut shader_name = shader_path
                .strip_prefix(&root_dir)
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned();
            if std::path::MAIN_SEPARATOR == '\\' {
                shader_name = shader_name.replace('\\', "/");
            }
            shader_set.insert(shader_name, Rc::new(shader));
        }
    }
}

impl Shader {
    pub fn new<P: AsRef<Path>>(device: &Device, spv_path: P) -> Self {
        let mut spv_file = File::open(&spv_path).unwrap();
        let spv_code = read_spv(&mut spv_file).unwrap();

        let shader_module = {
            let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_code);
            unsafe {
                device
                    .create_shader_module(&vertex_shader_info, None)
                    .unwrap()
            }
        };

        let mut shader = Self {
            stage: vk::ShaderStageFlags::VERTEX,
            descriptor_infos: Vec::new(),
            shader_module,
            use_descriptor_array: false,
            use_push_constants: false,
        };

        shader.parse_spv(&spv_code);

        shader
    }

    pub fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.shader_module, None);
        }
        self.shader_module = vk::ShaderModule::null();
    }

    /*
    ```glsl
    layout(binding = 0, set = 0)
    uniform PerFrame {
        mat4x4 pers_view_matrix;
    } per_frame;

    layout(binding = 0, set = 1)
    uniform sampler2D tex_sampler;
    ```

    Possible pseudo SPIR-V code of the glsl code snippet above:

    ```spirv
    ; annotations
    OpDecorate %18 Block

    ; types, variables, constants
    ; result_id is moved the right side of '=' for ease of reading.
    %6  = OpTypeFloat 32                  ; 32-bit float
    %7  = OpTypeVector %6 4               ; vec4
    %8  = OpTypeMatrix %7 4               ; mat4
    %18 = OpTypeStruct %8                 ; PerFrame
    %19 = OpTypePointer Uniform %18
    %20 = OpVariable %19 Uniform
    %21 = OpTypeSampler
    %22 = OpTypePointer Uniform %21
    %23 = OpVariable %22 Uniform
    ```
    */
    fn parse_spv(&mut self, spv_code: &Vec<u32>) {
        let id_bound = spv_code[3];
        let mut ids = vec![SpvId::default(); id_bound as usize];

        let mut stage_found = false;
        let mut word_pos = 5;
        while word_pos < spv_code.len() {
            let word = spv_code[word_pos];

            let opcode = (word & 0x0000_ffff) as u32;
            let word_count = (word >> 16) as usize;

            let instruction = &spv_code[word_pos..(word_pos + word_count)];

            let opcode = Op::from_u32(opcode).unwrap();
            match opcode {
                Op::EntryPoint => {
                    assert!(word_count >= 2);
                    let model = ExecutionModel::from_u32(instruction[1]).unwrap();
                    self.stage = shader_stage_from_execution_model(model);
                    stage_found = true;
                }
                Op::Decorate => {
                    assert!(word_count >= 3);
                    let target_id = instruction[1];
                    assert!(target_id < id_bound);
                    let decoration_type = Decoration::from_u32(instruction[2]).unwrap();
                    match decoration_type {
                        Decoration::DescriptorSet => {
                            assert!(word_count == 4);
                            ids[target_id as usize].set = instruction[3];
                        }
                        Decoration::Binding => {
                            assert!(word_count == 4);
                            ids[target_id as usize].binding = instruction[3];
                        }
                        _ => (),
                    }
                }
                Op::TypeStruct | Op::TypeImage | Op::TypeSampler | Op::TypeSampledImage => {
                    assert!(word_count >= 2);

                    let result_id = instruction[1];
                    assert!(result_id < id_bound);
                    let result_id = result_id as usize;

                    assert!(ids[result_id].opcode == Op::Nop);
                    ids[result_id].opcode = opcode;
                }
                Op::TypePointer => {
                    assert!(word_count == 4);

                    let result_id = instruction[1];
                    assert!(result_id < id_bound);
                    let result_id = result_id as usize;

                    assert!(ids[result_id].opcode == Op::Nop);
                    ids[result_id].opcode = opcode;
                    ids[result_id].storage_class = StorageClass::from_u32(instruction[2]).unwrap();
                    ids[result_id].type_id = instruction[3];
                }
                Op::Constant => {
                    assert!(word_count >= 4);

                    let result_id = instruction[2];
                    assert!(result_id < id_bound);
                    let result_id = result_id as usize;

                    assert!(ids[result_id].opcode == Op::Nop);
                    ids[result_id].opcode = opcode;
                    ids[result_id].type_id = instruction[1];
                    // Currently only support 32-bit constants.
                    ids[result_id].constant = instruction[3];
                }
                Op::Variable => {
                    assert!(word_count >= 4);

                    let result_id = instruction[2];
                    assert!(result_id < id_bound);
                    let result_id = result_id as usize;

                    assert!(ids[result_id].opcode == Op::Nop);
                    ids[result_id].opcode = opcode;
                    ids[result_id].type_id = instruction[1];
                    ids[result_id].storage_class = StorageClass::from_u32(instruction[3]).unwrap();
                }
                _ => (),
            }

            word_pos += word_count;
        }
        assert!(stage_found == true);

        for id in &ids {
            if id.opcode == Op::Variable {
                match id.storage_class {
                    StorageClass::Uniform
                    | StorageClass::UniformConstant
                    | StorageClass::StorageBuffer => {
                        assert!(ids[id.type_id as usize].opcode == Op::TypePointer);
                        let opcode = ids[ids[id.type_id as usize].type_id as usize].opcode;
                        let desc_type = descriptor_type_from_opcode(opcode);
                        self.descriptor_infos.push(DescriptorInfo {
                            set: id.set,
                            binding: id.binding,
                            kind: desc_type,
                        });
                    }
                    // StorageClass::UniformConstant => self.use_descriptor_array = true,
                    // StorageClass::PushConstant => self.use_push_constants = true,
                    _ => (),
                }
            }
        }
    }
}

impl Program {
    pub fn new(
        device: &Device,
        bind_point: vk::PipelineBindPoint,
        shaders: Vec<Rc<Shader>>,
    ) -> Self {
        let (desc_set_layouts, desc_set_layout_count) = {
            let mut set_bindings: [Vec<vk::DescriptorSetLayoutBinding>; 4] =
                [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

            for shader in &shaders {
                for desc_info in &shader.descriptor_infos {
                    let bindings = &mut set_bindings[desc_info.set as usize];
                    let mut binding = bindings.iter_mut().find(|b| b.binding == desc_info.binding);

                    // Check if a SetLayoutBinding with the same set index already exists. If so,
                    // that the current shader's stage to it, otherwise create a new
                    // SetLayoutBinding.
                    match binding.as_mut() {
                        Some(b) => {
                            assert!(b.descriptor_type == desc_info.kind);
                            b.stage_flags |= shader.stage;
                        }
                        None => {
                            set_bindings[desc_info.set as usize].push(
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(desc_info.binding)
                                    .descriptor_type(desc_info.kind)
                                    .descriptor_count(1)
                                    .stage_flags(shader.stage),
                            );
                        }
                    }
                }
            }
            let set_bindings_count = set_bindings.iter().filter(|sb| !sb.is_empty()).count();

            let mut desc_set_layouts = [vk::DescriptorSetLayout::null(); 4];
            let mut layout_count = 0;
            for (i, bindings) in set_bindings.iter().enumerate() {
                if bindings.len() > 0 {
                    let set_layout_createinfo =
                        vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
                    desc_set_layouts[i] = unsafe {
                        device
                            .create_descriptor_set_layout(&set_layout_createinfo, None)
                            .unwrap()
                    };
                    layout_count = i + 1;
                } else {
                    // Each index of desc_set_layouts represents a bindable descriptor set slot in the
                    // pipeline, there cannot be null inbetween valid set layouts.
                    break;
                }
            }
            assert!(set_bindings_count == layout_count);

            (desc_set_layouts, layout_count)
        };

        let pipeline_layout = {
            let createinfo = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&desc_set_layouts[0..desc_set_layout_count]);
            unsafe { device.create_pipeline_layout(&createinfo, None).unwrap() }
        };

        Self {
            shaders,
            bind_point,
            desc_set_layouts,
            desc_set_layout_count,
            pipeline_layout,
        }
    }

    pub fn desc_set_layouts(&self) -> &[vk::DescriptorSetLayout] {
        &self.desc_set_layouts[0..self.desc_set_layout_count]
    }

    pub fn destruct(&mut self, device: &Device) {
        for layout in self.desc_set_layouts.iter_mut() {
            if *layout != vk::DescriptorSetLayout::null() {
                unsafe {
                    device.destroy_descriptor_set_layout(*layout, None);
                }
            }
            *layout = vk::DescriptorSetLayout::null();
        }

        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
        self.pipeline_layout = vk::PipelineLayout::null();

        self.shaders.clear();
    }
}

fn shader_stage_from_execution_model(model: ExecutionModel) -> vk::ShaderStageFlags {
    match model {
        ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
        ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
        ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
        _ => panic!("Unhandled execution model for shader stage."),
    }
}

fn descriptor_type_from_opcode(op: Op) -> vk::DescriptorType {
    match op {
        // Op::TypeStruct => vk::DescriptorType::STORAGE_BUFFER,
        Op::TypeStruct => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        Op::TypeImage => vk::DescriptorType::STORAGE_IMAGE,
        Op::TypeSampledImage => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        Op::TypeSampler => vk::DescriptorType::SAMPLER,
        _ => panic!("Unhandled opcode for descriptor type."),
    }
}

impl Default for SpvId {
    fn default() -> Self {
        Self {
            opcode: Op::Nop,
            type_id: 0,
            storage_class: StorageClass::Generic,
            binding: 0,
            set: 0,
            constant: 0,
        }
    }
}
