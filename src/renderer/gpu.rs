use crate::log_info;

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
    pub pp_texture: (wgpu::Texture, wgpu::TextureView),
    pub pp_bind_group_layout: wgpu::BindGroupLayout,
    pub pp_bind_group: wgpu::BindGroup,
    pub pp_compute_pipeline: wgpu::ComputePipeline,
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

        let rc_shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../../res/shaders/rc.wgsl"));
        let rc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&cascades_bind_group_layout)],
            immediate_size: 12,
        });
        let rc_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&rc_pipeline_layout),
                module: &rc_shader_module,
                entry_point: Some("gen_cascades"),
                compilation_options: Default::default(),
                cache: None,
            });
        let merge_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&cascades_bind_group_layout)],
                immediate_size: 12,
            });
        let merge_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&merge_pipeline_layout),
                module: &rc_shader_module,
                entry_point: Some("merge_cascades"),
                compilation_options: Default::default(),
                cache: None,
            });
        let final_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&cascades_bind_group_layout)],
                immediate_size: 0,
            });
        let final_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&final_pipeline_layout),
                module: &rc_shader_module,
                entry_point: Some("final_pass"),
                compilation_options: Default::default(),
                cache: None,
            });

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
        let pp_texture_view = pp_texture.create_view(&Default::default());
        let pp_texture_bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: std::num::NonZeroU32::new(cascade_textures.len() as u32),
        };
        let pp_texture_bind_group_entry = wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&pp_texture_view),
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
            pp_texture: (pp_texture, pp_texture_view),
            pp_bind_group_layout,
            pp_bind_group,
            pp_compute_pipeline,
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
