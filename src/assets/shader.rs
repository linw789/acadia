use ash::{Device, util::read_spv, vk};
use spirv::{Decoration, ExecutionModel, Op, StorageClass};
use std::{fs::File, path::Path};
use std::{mem::transmute, vec::Vec};

pub struct Shader {
    name: String,
    stage: vk::ShaderStageFlags,
    descriptor_infos: Vec<DescriptorInfo>,
    shader_module: vk::ShaderModule,
    use_descriptor_array: bool,
    use_push_constants: bool,
}

pub struct Program {
    shaders: Vec<Shader>,
    bind_point: vk::PipelineBindPoint,
    set_layouts: [vk::DescriptorSetLayout; 4],
    descriptor_count: u32,
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
    storage_class: u32,
    binding: u32,
    set: u32,
    constant: u32,
}

impl Shader {
    pub(super) fn new<P: AsRef<Path>>(device: &Device, spv_path: P) -> Self {
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
            name: spv_path
                .as_ref()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned(),
            stage: vk::ShaderStageFlags::VERTEX,
            descriptor_infos: Vec::new(),
            shader_module,
            use_descriptor_array: false,
            use_push_constants: false,
        };

        shader.parse_spv(&spv_code);

        shader
    }

    pub(super) fn destruct(&mut self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.shader_module, None);
        }
        self.shader_module = vk::ShaderModule::null();
    }

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

            let opcode: Op = unsafe { transmute(opcode) };
            match opcode {
                Op::EntryPoint => {
                    assert!(word_count >= 2);
                    let model: ExecutionModel = unsafe { transmute(instruction[1]) };
                    self.stage = shader_stage_from_execution_model(model);
                    stage_found = true;
                }
                Op::Decorate => {
                    assert!(word_count >= 3);
                    let target_id = instruction[1];
                    assert!(target_id < id_bound);
                    let decoration_type: Decoration = unsafe { transmute(instruction[2]) };
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
                    ids[result_id].storage_class = instruction[2];
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
                    ids[result_id].storage_class = instruction[2];
                }
                _ => (),
            }

            word_pos += word_count;
        }
        assert!(stage_found == true);

        for id in &ids {
            if id.opcode == Op::Variable {
                match unsafe { transmute(id.storage_class) } {
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
    pub fn new(device: &Device, bind_point: vk::PipelineBindPoint, shaders: Vec<Shader>) -> Self {
        let set_layout = {
            let mut set_bindings: [Vec<vk::DescriptorSetLayoutBinding>; 4] =
                [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

            for shader in &shaders {
                for desc_info in shader.descriptor_infos {
                    let mut binding_index = -1;

                    let bindings = &set_bindings[desc_info.set as usize];
                    for (i, binding) in bindings.iter().enumerate() {
                        if binding.binding == desc_info.binding {
                            assert!(binding.descriptor_type == desc_info.kind);
                            binding_index = i as i32;
                        }
                    }

                    if binding_index > -1 {
                        bindings[binding_index as usize].stage_flags |= shader.stage;
                    } else {
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
        };

        Self {
            shaders,
            bind_point,
        }
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
        Op::TypeStruct => vk::DescriptorType::STORAGE_BUFFER,
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
            storage_class: 0,
            binding: 0,
            set: 0,
            constant: 0,
        }
    }
}
