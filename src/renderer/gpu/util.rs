use std::num::{NonZero, NonZeroU64};

/// Utility for creating storage / uniform buffers
pub struct Buffer {
    pub handle: wgpu::Buffer,
    pub bind_group_layout_entry: wgpu::BindGroupLayoutEntry,
}

pub struct BufferCreateInfo<'a> {
    pub label: Option<&'a str>,
    /// Binding index of the buffer in a bind group
    pub binding: u32,
    pub usage: wgpu::BufferUsages,
    pub visibility: wgpu::ShaderStages,
    pub buffer_binding_type: wgpu::BufferBindingType,
    pub has_dynamic_offset: bool,
    pub count: Option<NonZero<u32>>,
}

impl Buffer {
    /// Creates a buffer with size in bytes
    pub fn create(device: &wgpu::Device, buffer_create_info: BufferCreateInfo, size: u64) -> Self {
        let handle = device.create_buffer(&wgpu::BufferDescriptor {
            label: buffer_create_info.label,
            size,
            usage: buffer_create_info.usage,
            mapped_at_creation: false,
        });

        let bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: buffer_create_info.binding,
            visibility: buffer_create_info.visibility,
            ty: wgpu::BindingType::Buffer {
                ty: buffer_create_info.buffer_binding_type,
                has_dynamic_offset: buffer_create_info.has_dynamic_offset,
                min_binding_size: None,
            },
            count: buffer_create_info.count,
        };

        return Self {
            handle,
            bind_group_layout_entry,
        };
    }

    /// Creates a buffer with data
    ///
    /// # Notes
    /// * BufferDescriptor usage flags must contain wgpu::BufferUsages::COPY_DST
    pub fn create_with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_create_info: BufferCreateInfo,
        data: &[T],
        length: u64,
    ) -> Self {
        let handle = device.create_buffer(&wgpu::BufferDescriptor {
            label: buffer_create_info.label,
            size: length * size_of::<T>() as u64,
            usage: buffer_create_info.usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&handle, 0, bytemuck::cast_slice(data));

        let bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: buffer_create_info.binding,
            visibility: buffer_create_info.visibility,
            ty: wgpu::BindingType::Buffer {
                ty: buffer_create_info.buffer_binding_type,
                has_dynamic_offset: buffer_create_info.has_dynamic_offset,
                min_binding_size: Some(NonZeroU64::new(size_of::<T>() as u64)).unwrap(),
            },
            count: buffer_create_info.count,
        };

        return Self {
            handle,
            bind_group_layout_entry,
        };
    }

    pub fn bind_group_entry<'a>(&'a self) -> wgpu::BindGroupEntry<'a> {
        wgpu::BindGroupEntry {
            binding: self.bind_group_layout_entry.binding,
            resource: self.handle.as_entire_binding(),
        }
    }

    pub fn set_buffer_data<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        queue.write_buffer(&self.handle, 0, bytemuck::cast_slice(data));
    }
}

pub struct ComputePipelineCreateInfo<'a> {
    pub label: Option<&'a str>,
    pub shader_module: &'a wgpu::ShaderModule,
    pub entry_point: Option<&'a str>,
    pub bind_group_layouts: &'a [Option<&'a wgpu::BindGroupLayout>],
    pub immediate_size: u32,
}

pub fn create_compute_pipeline(
    device: &wgpu::Device,
    create_info: ComputePipelineCreateInfo,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: create_info.label,
        bind_group_layouts: create_info.bind_group_layouts,
        immediate_size: create_info.immediate_size,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: create_info.label,
        layout: Some(&pipeline_layout),
        module: create_info.shader_module,
        entry_point: create_info.entry_point,
        compilation_options: Default::default(),
        cache: None,
    });

    return pipeline;
}
