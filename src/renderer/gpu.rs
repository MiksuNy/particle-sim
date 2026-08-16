use crate::{game::Particle, log_info};

mod util;

pub struct GpuState {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub cascades_bind_group_layout: wgpu::BindGroupLayout,
    pub cascades_bind_group: wgpu::BindGroup,
    pub cascade_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    pub rc_compute_pipeline: wgpu::ComputePipeline,
    pub merge_compute_pipeline: wgpu::ComputePipeline,
    pub final_compute_pipeline: wgpu::ComputePipeline,
    pub pp_texture: wgpu::Texture,
    pub pp_bind_group_layout: wgpu::BindGroupLayout,
    pub pp_bind_group: wgpu::BindGroup,
    pub pp_compute_pipeline: wgpu::ComputePipeline,
    pub particle_storage_buffer: util::Buffer,
    pub world_bind_group: wgpu::BindGroup,
}

impl GpuState {
    pub fn new(dimensions: (u32, u32), num_cascades: u32) -> Self {
        let (instance, adapter) = Self::get_instance_and_adapter();
        let (device, queue) = Self::create_device_and_queue(&adapter);

        let mut cascade_textures: Vec<(wgpu::Texture, wgpu::TextureView)> = Vec::new();
        for _ in 0..num_cascades {
            let cascade_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: dimensions.0,
                    height: dimensions.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let cascade_texture_view = cascade_texture.create_view(&Default::default());
            cascade_textures.push((cascade_texture, cascade_texture_view));
        }

        let cascade_textures_bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: std::num::NonZeroU32::new(cascade_textures.len() as u32),
        };
        let cascade_texture_views = cascade_textures
            .iter()
            .map(|(_, view)| view)
            .collect::<Vec<&wgpu::TextureView>>();
        let cascade_textures_bind_group_entry = wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureViewArray(&cascade_texture_views),
        };

        let cascades_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[cascade_textures_bind_group_layout_entry],
            });
        let cascades_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cascades_bind_group_layout,
            entries: &[cascade_textures_bind_group_entry.clone()],
        });

        let particle_storage_buffer = util::Buffer::create_with_data::<GpuParticle>(
            &device,
            &queue,
            util::BufferCreateInfo {
                label: None,
                binding: 0,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                visibility: wgpu::ShaderStages::COMPUTE,
                buffer_binding_type: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                count: None,
            },
            &[],
            32,
        );

        let world_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[particle_storage_buffer.bind_group_layout_entry],
            });
        let world_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &world_bind_group_layout,
            entries: &[particle_storage_buffer.bind_group_entry()],
        });

        let rc_shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../../res/shaders/rc.wgsl"));

        let rc_compute_pipeline = util::create_compute_pipeline(
            &device,
            util::ComputePipelineCreateInfo {
                label: None,
                shader_module: &rc_shader_module,
                entry_point: Some("gen_cascades"),
                bind_group_layouts: &[
                    Some(&cascades_bind_group_layout),
                    Some(&world_bind_group_layout),
                ],
                immediate_size: 12,
            },
        );
        let merge_compute_pipeline = util::create_compute_pipeline(
            &device,
            util::ComputePipelineCreateInfo {
                label: None,
                shader_module: &rc_shader_module,
                entry_point: Some("merge_cascades"),
                bind_group_layouts: &[Some(&cascades_bind_group_layout)],
                immediate_size: 12,
            },
        );
        let final_compute_pipeline = util::create_compute_pipeline(
            &device,
            util::ComputePipelineCreateInfo {
                label: None,
                shader_module: &rc_shader_module,
                entry_point: Some("final_pass"),
                bind_group_layouts: &[Some(&cascades_bind_group_layout)],
                immediate_size: 0,
            },
        );

        let pp_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pp_texture_bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let pp_texture_bind_group_entry = wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(
                &pp_texture.create_view(&Default::default()),
            ),
        };
        let pp_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    cascade_textures_bind_group_layout_entry,
                    pp_texture_bind_group_layout_entry,
                ],
            });
        let pp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pp_bind_group_layout,
            entries: &[
                cascade_textures_bind_group_entry,
                pp_texture_bind_group_entry,
            ],
        });
        let pp_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&pp_bind_group_layout)],
            immediate_size: 0,
        });
        let pp_shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../../res/shaders/pp.wgsl"));
        let pp_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pp_pipeline_layout),
                module: &pp_shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        return Self {
            instance,
            adapter,
            device,
            queue,
            cascades_bind_group_layout,
            cascades_bind_group,
            cascade_textures,
            rc_compute_pipeline,
            merge_compute_pipeline,
            final_compute_pipeline,
            pp_texture,
            pp_bind_group_layout,
            pp_bind_group,
            pp_compute_pipeline,
            particle_storage_buffer,
            world_bind_group,
        };
    }

    fn get_instance_and_adapter() -> (wgpu::Instance, wgpu::Adapter) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("Failed to create adapter");
        log_info!("{:#?}", adapter.get_info());
        return (instance, adapter);
    }

    fn create_device_and_queue(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        if !downlevel_capabilities
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            panic!("Adapter does not support compute shaders");
        }

        return pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                | wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                | wgpu::Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                | wgpu::Features::IMMEDIATES,
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("Failed to create device");
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct GpuParticle {
    color: glam::Vec4,
    pos: glam::Vec2,
    radius: f32,
    _pad: [u8; 4],
}

impl From<&Particle> for GpuParticle {
    fn from(value: &Particle) -> Self {
        Self {
            color: value.color,
            pos: value.pos,
            radius: value.radius,
            _pad: [0; 4],
        }
    }
}
