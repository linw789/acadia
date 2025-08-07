use ::ash::vk::{MemoryPropertyFlags, MemoryRequirements, PhysicalDeviceMemoryProperties};

pub fn find_memorytype_index(
    memory_req: &MemoryRequirements,
    memory_prop: &PhysicalDeviceMemoryProperties,
    flags: MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            ((1 << index) & memory_req.memory_type_bits) != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}
