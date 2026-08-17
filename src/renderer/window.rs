use std::{cell::RefCell, collections::HashSet, rc::Rc, sync::Arc};

use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop, OwnedDisplayHandle},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::{
    game::{Particle, World},
    log_info,
    renderer::gpu::{GpuParticle, GpuState},
};

struct AppState {
    world: Rc<RefCell<World>>,
    options: Rc<RefCell<AppOptions>>,
    gpu_state: GpuState,
    window: Arc<Window>,
    size: PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    surface_texture: wgpu::Texture,
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    key_states: HashSet<PhysicalKey>,
    cursor_position: PhysicalPosition<u32>,
}

impl AppState {
    async fn new(_display: OwnedDisplayHandle, window: Arc<Window>, options: AppOptions) -> Self {
        let gpu_state = GpuState::new(options.window_dimensions, options.num_cascades);
        let _ = window.request_inner_size(PhysicalSize::new(
            options.window_dimensions.0,
            options.window_dimensions.1,
        ));
        let size = window.inner_size();

        let surface = gpu_state.instance.create_surface(window.clone()).unwrap();
        let surface_capabilities = surface.get_capabilities(&gpu_state.adapter);
        let surface_format = surface_capabilities.formats[0];

        let shader_module = gpu_state
            .device
            .create_shader_module(wgpu::include_wgsl!("../../res/shaders/main.wgsl"));

        let bind_group_layout =
            gpu_state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });

        let surface_texture = gpu_state.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: options.window_dimensions.0,
                height: options.window_dimensions.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let surface_texture_view =
            surface_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let surface_texture_sampler = gpu_state.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let bind_group = gpu_state
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&surface_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&surface_texture_sampler),
                    },
                ],
            });

        let render_pipeline_layout =
            gpu_state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let render_pipeline =
            gpu_state
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader_module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader_module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(surface_format.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                });

        let world = World::new(glam::Vec2::new(
            options.window_dimensions.0 as f32,
            options.window_dimensions.1 as f32,
        ));

        let app = Self {
            world: Rc::new(RefCell::new(world)),
            options: Rc::new(RefCell::new(options)),
            gpu_state,
            window,
            size,
            surface,
            surface_format,
            surface_texture,
            render_pipeline,
            bind_group,
            key_states: HashSet::new(),
            cursor_position: PhysicalPosition { x: 0, y: 0 },
        };
        app.configure_surface();
        return app;
    }

    fn get_window(&self) -> &Window {
        &self.window
    }

    fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: self.surface_format,
            view_formats: vec![],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.width,
            height: self.size.height,
            desired_maximum_frame_latency: 1,
            present_mode: wgpu::PresentMode::AutoVsync,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        self.surface
            .configure(&self.gpu_state.device, &surface_config);
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.configure_surface();
    }

    fn render(&mut self) {
        let gpu_state = &mut self.gpu_state;
        let options = self.options.borrow();

        let surface_texture: wgpu::SurfaceTexture;
        let surface_texture_view = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => {
                surface_texture = texture;
                surface_texture.texture.create_view(&Default::default())
            }
            _ => panic!("Failed to get next surface texture"),
        };

        let mut command_encoder = gpu_state.device.create_command_encoder(&Default::default());

        // Generate sdf texture pass
        {
            let mut gen_sdf_texture_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            gen_sdf_texture_pass.set_bind_group(0, &gpu_state.cascades_bind_group, &[]);
            gen_sdf_texture_pass.set_bind_group(1, &gpu_state.world_bind_group, &[]);
            gen_sdf_texture_pass.set_pipeline(&gpu_state.gen_sdf_compute_pipeline);
            gen_sdf_texture_pass.dispatch_workgroups(
                options.window_dimensions.0 / 8,
                options.window_dimensions.1 / 8,
                1,
            );
        }

        // Generate cascades pass
        for cascade_index in 0..options.num_cascades {
            let mut rc_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            rc_pass.set_bind_group(0, &gpu_state.cascades_bind_group, &[]);
            rc_pass.set_bind_group(1, &gpu_state.world_bind_group, &[]);
            rc_pass.set_pipeline(&gpu_state.rc_compute_pipeline);
            let immediates = [
                cascade_index.to_le_bytes(),
                self.cursor_position.x.to_le_bytes(),
                self.cursor_position.y.to_le_bytes(),
            ];
            let bytes = immediates.as_flattened();
            rc_pass.set_immediates(0, bytes);
            rc_pass.dispatch_workgroups(
                options.window_dimensions.0 / 8,
                options.window_dimensions.1 / 8,
                1,
            );
        }

        // Merge cascades pass
        // Go from the lowest spatial resolution cascade to the lowest angular resolution
        // cascade (cascade 0)
        for cascade_index in (0..options.num_cascades - 1).rev() {
            let mut merge_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            merge_pass.set_bind_group(0, &gpu_state.cascades_bind_group, &[]);
            merge_pass.set_pipeline(&gpu_state.merge_compute_pipeline);
            let immediates = [
                cascade_index.to_le_bytes(),
                self.cursor_position.x.to_le_bytes(),
                self.cursor_position.y.to_le_bytes(),
            ];
            let bytes = immediates.as_flattened();
            merge_pass.set_immediates(0, bytes);
            merge_pass.dispatch_workgroups(
                options.window_dimensions.0 / 8,
                options.window_dimensions.1 / 8,
                1,
            );
        }

        // Final pass
        // Bilinearly interpolate cascade 0 probes
        {
            let mut final_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            final_pass.set_bind_group(0, &gpu_state.cascades_bind_group, &[]);
            final_pass.set_pipeline(&gpu_state.final_compute_pipeline);
            final_pass.dispatch_workgroups(
                options.window_dimensions.0 / 8,
                options.window_dimensions.1 / 8,
                1,
            );
        }

        // Post-process pass
        {
            let mut pp_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pp_pass.set_bind_group(0, &gpu_state.pp_bind_group, &[]);
            pp_pass.set_pipeline(&gpu_state.pp_compute_pipeline);
            pp_pass.dispatch_workgroups(
                options.window_dimensions.0 / 8,
                options.window_dimensions.1 / 8,
                1,
            );
        }

        // Copy the result of post processing pass to surface texture
        command_encoder.copy_texture_to_texture(
            gpu_state.pp_texture.as_image_copy(),
            self.surface_texture.as_image_copy(),
            self.surface_texture.size(),
        );

        // Render pass
        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw(0..3, 0..1);
        }

        gpu_state.queue.submit([command_encoder.finish()]);
        self.window.pre_present_notify();
        gpu_state.queue.present(surface_texture);
    }
}

#[derive(Clone, Copy)]
struct AppOptions {
    window_dimensions: (u32, u32),
    num_cascades: u32,
}

impl Default for AppOptions {
    fn default() -> Self {
        Self {
            window_dimensions: (512, 512),
            num_cascades: 6,
        }
    }
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let app_state = pollster::block_on(AppState::new(
            event_loop.owned_display_handle(),
            window.clone(),
            AppOptions::default(),
        ));

        self.state = Some(app_state);

        window.set_title("Epic 2D particle sim");

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log_info!("Closing window");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let app_state = self.state.as_mut().unwrap();

                app_state.world.borrow_mut().update(1.0 / 20.0);

                let gpu_particles: Vec<GpuParticle> = app_state
                    .world
                    .borrow_mut()
                    .particles
                    .iter()
                    .map(|particle| particle.into())
                    .collect();
                app_state
                    .gpu_state
                    .particle_storage_buffer
                    .set_buffer_data(&app_state.gpu_state.queue, &gpu_particles);

                app_state.render();
                app_state.get_window().request_redraw();
            }
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {
                let app_state = self.state.as_mut().unwrap();
                app_state.cursor_position = position.cast();
            }
            WindowEvent::Resized(new_size) => {
                self.state.as_mut().unwrap().resize(new_size);
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::Key(key) => match key.physical_key {
                PhysicalKey::Code(KeyCode::Space) => {
                    if key.state.is_pressed() {
                        let app_state = self.state.as_mut().unwrap();
                        app_state.world.borrow_mut().add_particle(Particle::new(
                            glam::Vec2::new(
                                app_state.cursor_position.x as f32,
                                app_state.cursor_position.y as f32,
                            ),
                            5.0,
                        ));
                    }
                }
                _ => (),
            },
            _ => (),
        }
    }
}

pub fn start_window() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
