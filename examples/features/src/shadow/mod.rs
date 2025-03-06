use std::{f32::consts, iter, mem::size_of, ops::Range, sync::Arc};
use std::ops::Deref;
use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Quat, Vec3};
use nanorand::Rng;
use wgpu::{BufferAddress, Features, PolygonMode};
use wgpu::util::{align_to, DeviceExt, DrawIndexedIndirectArgs, DrawIndirectArgs};

const WIREFRAME : bool = false;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _normal: [f32; 4],
}

fn vertex(pos: [i8; 3], nor: [i8; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _normal: [nor[0] as f32, nor[1] as f32, nor[2] as f32, 0.0],
    }
}

fn vertexf(pos: [f32; 3], nor: [f32; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1.0],
        _normal: [nor[0], nor[1], nor[2], 0.0],
    }
}

const GRASS_COUNT : u32 = 500u32;

fn create_cube() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0, 1]),
        vertex([1, -1, 1], [0, 0, 1]),
        vertex([1, 1, 1], [0, 0, 1]),
        vertex([-1, 1, 1], [0, 0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [0, 0, -1]),
        vertex([1, 1, -1], [0, 0, -1]),
        vertex([1, -1, -1], [0, 0, -1]),
        vertex([-1, -1, -1], [0, 0, -1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [1, 0, 0]),
        vertex([1, 1, -1], [1, 0, 0]),
        vertex([1, 1, 1], [1, 0, 0]),
        vertex([1, -1, 1], [1, 0, 0]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [-1, 0, 0]),
        vertex([-1, 1, 1], [-1, 0, 0]),
        vertex([-1, 1, -1], [-1, 0, 0]),
        vertex([-1, -1, -1], [-1, 0, 0]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [0, 1, 0]),
        vertex([-1, 1, -1], [0, 1, 0]),
        vertex([-1, 1, 1], [0, 1, 0]),
        vertex([1, 1, 1], [0, 1, 0]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, -1, 0]),
        vertex([-1, -1, 1], [0, -1, 0]),
        vertex([-1, -1, -1], [0, -1, 0]),
        vertex([1, -1, -1], [0, -1, 0]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

fn create_plane(size: i8) -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        vertex([size, -size, 0], [0, 0, 1]),
        vertex([size, size, 0], [0, 0, 1]),
        vertex([-size, -size, 0], [0, 0, 1]),
        vertex([-size, size, 0], [0, 0, 1]),
    ];

    let index_data: &[u16] = &[0, 1, 2, 2, 1, 3];

    (vertex_data.to_vec(), index_data.to_vec())
}

fn create_grass_blade(baseWidth: f32, height: f32, steps: u16) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertex_data : Vec<Vertex> = vec![];
    let mut index_data : Vec<u16> = vec![];

    let steps = steps.max(2);
    // all the steps

    let baseWidthVec = glam::vec3(baseWidth * 0.5, 0.0, 0.0); // halved to centre around 0.0
    let heightVec = glam::vec3(0.0, height, 0.0);

    let normal: [f32; 3] = [0.0, 0.0, 1.0];

    // first step
    vertex_data.push(vertexf([baseWidthVec.x, 0.0, 0.0], normal.clone()));
    vertex_data.push(vertexf([-baseWidthVec.x, 0.0, 0.0], normal.clone()));

    for i in (1..=steps) {

        let newPoint = baseWidthVec.lerp(heightVec, i as f32 / steps as f32);

        // we want to order data such that in a pack of 4, the vertices at the top are always at the end
        if (i < steps)
        {
            vertex_data.push(vertexf([newPoint.x, newPoint.y, 0.0], normal.clone()));
            vertex_data.push(vertexf([-newPoint.x, newPoint.y, 0.0], normal.clone()));

            index_data.append([1, 0, 2, 2, 3, 1].map(|x| { x + (2 * (i - 1)) }).to_vec().as_mut());
        }
        else
        {
            // last step
            vertex_data.push(vertexf([0.0, newPoint.y, 0.0], normal.clone()));
            vertex_data.push(vertexf([0.0, newPoint.y, 0.0], normal.clone()));

            index_data.append([1, 0, 2].map(|x| { x + 2 * (i - 1) }).to_vec().as_mut());
        }

    }

    (vertex_data, index_data)
}


struct Entity {
    mx_world: glam::Mat4,
    rotation_speed: f32,
    color: wgpu::Color,
    const_vertex_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: Arc<wgpu::Buffer>,
    index_format: wgpu::IndexFormat,
    index_count: usize,
    uniform_offset: wgpu::DynamicOffset,
}

struct Light {
    pos: glam::Vec3,
    color: wgpu::Color,
    fov: f32,
    depth: Range<f32>,
    target_view: wgpu::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightRaw {
    proj: [[f32; 4]; 4],
    pos: [f32; 4],
    color: [f32; 4],
}

impl Light {
    fn to_raw(&self) -> LightRaw {
        let view = glam::Mat4::look_at_rh(self.pos, glam::Vec3::ZERO, glam::Vec3::Z);
        let projection = glam::Mat4::perspective_rh(
            self.fov * consts::PI / 180.,
            1.0,
            self.depth.start,
            self.depth.end,
        );
        let view_proj = projection * view;
        LightRaw {
            proj: view_proj.to_cols_array_2d(),
            pos: [self.pos.x, self.pos.y, self.pos.z, 1.0],
            color: [
                self.color.r as f32,
                self.color.g as f32,
                self.color.b as f32,
                1.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalUniforms {
    proj: [[f32; 4]; 4],
    num_lights: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EntityUniforms {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}
// All the entities
// #[repr(C)]
// #[derive(Clone, Pod, Zeroable)]
// struct EntityBuffer {
//     entities: Vec<EntityUniforms>,
// }

struct Pass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
}

struct ComputePass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    storage_buf: wgpu::Buffer,
    vertex_size: wgpu::BufferAddress,
}

struct Example {
    entities: Vec<Entity>,
    lights: Vec<Light>,
    lights_are_dirty: bool,
    compute_pass: ComputePass,
    shadow_pass: Pass,
    forward_pass: Pass,
    forward_depth: wgpu::TextureView,
    entity_bind_group: wgpu::BindGroup,
    wind_bind_group: wgpu::BindGroup,
    grass_wind_uniform_buf: wgpu::Buffer,
    light_storage_buf: wgpu::Buffer,
    entity_uniform_buf: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    now: web_time::Instant,
    config: wgpu::SurfaceConfiguration,
    wind_update_counter: u32
}

impl Example {
    const MAX_LIGHTS: usize = 10;
    const VERTS_PER_GRASSBLADE : u16 = 128; // needs to be an even number, using 2x compute shader workgroup size for now so we can do 1 grass blade per invocation
    const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    const SHADOW_SIZE: wgpu::Extent3d = wgpu::Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: Self::MAX_LIGHTS as u32,
    };
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn generate_matrix(aspect_ratio: f32, elapsed_secs: f64) -> glam::Mat4 {
        let projection = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 1.0, 200.0);
        let spin_radius = 10.0;
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(spin_radius * elapsed_secs.cos() as f32, spin_radius * elapsed_secs.sin() as f32, 10.0),
            glam::Vec3::new(0f32, 0.0, 0.0),
            glam::Vec3::Z,
        );
        projection * view
    }

    fn create_depth_texture(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
            view_formats: &[],
        });

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

impl crate::framework::Example for Example {
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults() // These downlevel limits will allow the code to run on all possible hardware
    }

    fn required_features() -> wgpu::Features {
        Features::MULTI_DRAW_INDIRECT
    }

    fn optional_features() -> wgpu::Features {
        wgpu::Features::DEPTH_CLIP_CONTROL | wgpu::Features::POLYGON_MODE_LINE
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let supports_storage_resources = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::VERTEX_STORAGE)
            && device.limits().max_storage_buffers_per_shader_stage > 0;

        // Create the vertex and index buffers
        let vertex_size = size_of::<Vertex>()  as wgpu::BufferAddress;
        let vbo_size =
            vertex_size * Self::VERTS_PER_GRASSBLADE as wgpu::BufferAddress;
        // NOTE KP: JANK HARDCODE, FIX STEP CALCULATION WRT VERTS PER GRASSBLADE LATER
        let (vbo_vertex_data, cube_index_data) = create_grass_blade(0.5, 2.0, (Self::VERTS_PER_GRASSBLADE as u16 / 2) - 1);


        let cube_index_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Cubes Index Buffer"),
                contents: bytemuck::cast_slice(&cube_index_data),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));

        let (plane_vertex_data, plane_index_data) = create_plane(100);

        let plane_vertex_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Plane Vertex Buffer"),
            contents: bytemuck::cast_slice(&plane_vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        };
        let plane_vertex_buf = device.create_buffer_init(&plane_vertex_desc);
        let const_plane_vertex_buf = device.create_buffer_init(&plane_vertex_desc);


        let plane_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plane Index Buffer"),
            contents: bytemuck::cast_slice(&plane_index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        struct CubeDesc {
            offset: glam::Vec3,
            angle: f32,
            scale: f32,
            rotation: f32,
        }
        let cube_desc = CubeDesc {
                offset: glam::Vec3::new(0.0, 0.0, 0.0),
                angle: 10.0,
                scale: 0.7,
                rotation: 0.0,
            };

        let entity_uniform_size = size_of::<EntityUniforms>() as wgpu::BufferAddress;
        let num_entities = 2u64; // plane + grass
        // Make the `uniform_alignment` >= `entity_uniform_size` and aligned to `min_uniform_buffer_offset_alignment`.
        let uniform_alignment = {
            let alignment =
                device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
            align_to(entity_uniform_size, alignment)
        };
        // Note: dynamic uniform offsets also have to be aligned to `Limits::min_uniform_buffer_offset_alignment`.
        // Note KP: We will use this buffer for both storage and dynamic offset binding!
        let entity_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: num_entities * uniform_alignment,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_format = wgpu::IndexFormat::Uint16;

        let mut entities = vec![{
            // plane
            Entity {
                mx_world: glam::Mat4::from_scale_rotation_translation(
                    Vec3::ONE,
                    Quat::IDENTITY,
                    Vec3::new(0.0, 0.0, 0.0)),
                rotation_speed: 0.0,
                color: wgpu::Color::WHITE,
                const_vertex_buf: const_plane_vertex_buf,
                vertex_buf: plane_vertex_buf,
                index_buf: Arc::new(plane_index_buf),
                index_format,
                index_count: plane_index_data.len(),
                uniform_offset: 0,
            }
        }];


        let mx_world = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(cube_desc.scale),
            glam::Quat::from_euler(EulerRot::XYZ, consts::PI / 2.0, consts::PI / 4.0, 0.0),
            cube_desc.offset,
        );

        let cube_vertex_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(format!("Grass Vertex Buffer").as_str()),
                contents: bytemuck::cast_slice(&vbo_vertex_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            },
        );

        let runtime_vertex_buf = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some(format!("Grass Runtime Vertex Buffer").as_str()),
                size: vbo_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );

        entities.push(Entity {
            mx_world,
            rotation_speed: cube_desc.rotation,
            color: wgpu::Color::GREEN,
            const_vertex_buf: cube_vertex_buf,
            vertex_buf: runtime_vertex_buf,
            index_buf: Arc::clone(&cube_index_buf),
            index_format,
            index_count: cube_index_data.len(),
            uniform_offset: (1 * uniform_alignment as u32) as _,
        });

        let mut rng = nanorand::WyRand::new();
        let mut spread_offsets : Vec<[f32; 2]> = Vec::new();
        let mut wind_offsets : Vec<[f32; 2]> = Vec::new();
        let offset_magnitude = 5.0;
        for _ in 0..GRASS_COUNT {
            spread_offsets.push([
                (rng.generate::<f32>() * 2.0 - 1.0) * offset_magnitude, 
                (rng.generate::<f32>() * 2.0 - 1.0) * offset_magnitude,
            ]);
            wind_offsets.push([
                (rng.generate::<f32>() * 2.0 - 1.0), 
                (rng.generate::<f32>() * 2.0 - 1.0),
            ]);
        }

        // currently only grass in the indirect buffer
        // one "patch" of grass
        let grass_patch_draw_args = wgpu::util::DrawIndexedIndirectArgs{
            index_count: cube_index_data.len() as u32,
            instance_count: GRASS_COUNT as u32,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };

        // let mut indirect_bytes = Vec::new();

        // for i in 0..(entities.len() - 1) {
        //     indirect_bytes.extend_from_slice(indirect_args.as_bytes());
        // }

        let grass_patch_draw_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Indirect Buffer"),
                contents: grass_patch_draw_args.as_bytes(),
                usage: wgpu::BufferUsages::INDIRECT,
            },
        );

        let entity_mx_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(entity_uniform_size),
                    },
                    count: None,
                }],
                label: None,
            });
        let entity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &entity_mx_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &entity_uniform_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(entity_uniform_size),
                }),
            }],
            label: None,
        });

        // Create other resources
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: Self::SHADOW_SIZE,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut shadow_target_views = (0..2)
            .map(|i| {
                Some(shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow"),
                    format: None,
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                }))
            })
            .collect::<Vec<_>>();
        let lights = vec![
            Light {
                pos: glam::Vec3::new(7.0, -5.0, 10.0),
                color: wgpu::Color {
                    r: 0.5,
                    g: 1.0,
                    b: 0.5,
                    a: 1.0,
                },
                fov: 60.0,
                depth: 0.001..10000.0,
                target_view: shadow_target_views[0].take().unwrap(),
            }
            // Light {
            //     pos: glam::Vec3::new(-5.0, 7.0, 10.0),
            //     color: wgpu::Color {
            //         r: 1.0,
            //         g: 0.5,
            //         b: 0.5,
            //         a: 1.0,
            //     },
            //     fov: 45.0,
            //     depth: 1.0..20.0,
            //     target_view: shadow_target_views[1].take().unwrap(),
            // },
        ];
        let light_uniform_size = (Self::MAX_LIGHTS * size_of::<LightRaw>()) as wgpu::BufferAddress;
        let light_storage_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: light_uniform_size,
            usage: if supports_storage_resources {
                wgpu::BufferUsages::STORAGE
            } else {
                wgpu::BufferUsages::UNIFORM
            } | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_attr = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        let vb_desc = wgpu::VertexBufferLayout {
            array_stride: vertex_size,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attr,
        };

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let now = web_time::Instant::now();

        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32, now.elapsed().as_secs_f64());

        let forward_uniforms = GlobalUniforms {
            proj: mx_total.to_cols_array_2d(),
            num_lights: [lights.len() as u32, 0, 0, 0],
        };

        let view_proj_globals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&forward_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let grass_vbo_buffer_size =
            vbo_size * GRASS_COUNT as wgpu::BufferAddress;
        let grass_vbo_storage_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grass VBO"),
            size: grass_vbo_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST |  wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });


        let grass_spread_uniform_buf_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Spread Offset Buffer"),
            contents: bytemuck::cast_slice(&spread_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        };
        let grass_spread_uniform_buf = device.create_buffer_init(&grass_spread_uniform_buf_desc);

        let grass_wind_uniform_buf_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Wind Offset Buffer"),
            contents: bytemuck::cast_slice(&wind_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        };
        let grass_wind_uniform_buf = device.create_buffer_init(&grass_wind_uniform_buf_desc);


        let wind_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, // spread offsets
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size_of::<[f32; 2]>() as wgpu::BufferAddress)
                },
                count: None,
            }],
            label: None,
        });
        let wind_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &wind_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grass_wind_uniform_buf.as_entire_binding(),
            }],
            label: None,
        });


        let compute_pass = {
            let uniform_size = size_of::<GlobalUniforms>() as wgpu::BufferAddress;

            // Create pipeline layout
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0, // vbo storage
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(vertex_size)
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1, // uniforms storage
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(entity_uniform_size)
                        },
                        count: None,
                    }],
                });


            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow"),
                bind_group_layouts: &[&bind_group_layout, &wind_bind_group_layout],
                push_constant_ranges: &[],
            });

            // we need to allocate: model + color + vertex buffer (calculated as vertex * vert per grassblade) for each grass blade


            // Create bind group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grass_vbo_storage_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: entity_uniform_buf.as_entire_binding(),
                }],
                label: None,
            });
            

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
                label: Some("Compute"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("bezier_offset"),
                compilation_options: Default::default(),
                cache: None,
            });

            ComputePass {
                pipeline,
                bind_group,
                storage_buf: grass_vbo_storage_buf.clone(),
                vertex_size: vertex_size,
            }
        };

        let shadow_pass = {
            let uniform_size = size_of::<GlobalUniforms>() as wgpu::BufferAddress;
            // Create pipeline layout
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0, // global
                        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(uniform_size),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4, // spread offsets
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_of::<[f32; 2]>() as wgpu::BufferAddress),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5, // grass vbo storage
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(vertex_size)
                        },
                        count: None,
                    }],
                });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow"),
                bind_group_layouts: &[&bind_group_layout, &entity_mx_bind_group_layout],
                push_constant_ranges: &[],
            });

            let shadow_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: uniform_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create bind group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grass_spread_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grass_vbo_storage_buf.as_entire_binding(),
                }],
                label: None,
            });

            // Create the render pipeline
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_bake"),
                    compilation_options: Default::default(),
                    buffers: &[vb_desc.clone()],
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: device
                        .features()
                        .contains(wgpu::Features::DEPTH_CLIP_CONTROL),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::SHADOW_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2, // corresponds to bilinear filtering
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            Pass {
                pipeline,
                bind_group,
                uniform_buf: shadow_uniforms,
            }
        };

        let forward_pass = {
            // Create pipeline layout
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0, // global
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    size_of::<GlobalUniforms>() as _,
                                ),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1, // lights
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: if supports_storage_resources {
                                    wgpu::BufferBindingType::Storage { read_only: true }
                                } else {
                                    wgpu::BufferBindingType::Uniform
                                },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(light_uniform_size),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4, // spread offsets
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(size_of::<[f32; 2]>() as wgpu::BufferAddress),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5, // grass vbo storage
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(vertex_size)
                            },
                            count: None,
                        },
                    ],
                    label: None,
                });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("main"),
                bind_group_layouts: &[&bind_group_layout, &entity_mx_bind_group_layout],
                push_constant_ranges: &[],
            });


            // Create bind group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: view_proj_globals_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: light_storage_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: grass_spread_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: grass_vbo_storage_buf.as_entire_binding(),
                    },
                ],
                label: None,
            });

            // Create the render pipeline
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("main"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[vb_desc],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if supports_storage_resources {
                        "fs_main"
                    } else {
                        "fs_main_without_storage"
                    }),
                    compilation_options: Default::default(),
                    targets: &[Some(config.view_formats[0].into())],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: 
                        if WIREFRAME && device.features().contains(wgpu::Features::POLYGON_MODE_LINE) { 
                            PolygonMode::Line 
                        } else { 
                            PolygonMode::Fill 
                        },
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            Pass {
                pipeline,
                bind_group,
                uniform_buf: view_proj_globals_buf,
            }
        };

        let forward_depth = Self::create_depth_texture(config, device);

        Example {
            entities,
            lights,
            lights_are_dirty: true,
            compute_pass,
            shadow_pass,
            forward_pass,
            forward_depth,
            light_storage_buf,
            entity_uniform_buf,
            entity_bind_group,
            wind_bind_group,
            grass_wind_uniform_buf,
            indirect_buffer: grass_patch_draw_buf,
            now,
            config: config.clone(),
            wind_update_counter: 0
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        // update view-projection matrix
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32, self.now.elapsed().as_secs_f64());
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(
            &self.forward_pass.uniform_buf,
            0,
            bytemuck::cast_slice(mx_ref),
        );

        self.forward_depth = Self::create_depth_texture(config, device);
        self.config = config.clone();
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // update view-projection matrix
        let camera_speed = 0.4;
        let mx_total = Self::generate_matrix(self.config.width as f32 / self.config.height as f32,self.now.elapsed().as_secs_f64() * camera_speed);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(
            &self.forward_pass.uniform_buf,
            0,
            bytemuck::cast_slice(mx_ref),
        );



        //update uniforms
        for entity in self.entities.iter_mut() {
            if entity.rotation_speed != 0.0 {
                let rotation =
                    glam::Mat4::from_rotation_x(entity.rotation_speed * consts::PI / 180.);
                entity.mx_world *= rotation;
            }
            let data = EntityUniforms {
                model: entity.mx_world.to_cols_array_2d(),
                color: [
                    entity.color.r as f32,
                    entity.color.g as f32,
                    entity.color.b as f32,
                    entity.color.a as f32,
                ],
            };
            queue.write_buffer(
                &self.entity_uniform_buf,
                entity.uniform_offset as wgpu::BufferAddress,
                bytemuck::bytes_of(&data),
            );

            // write grass
        }

        let wind_update_freq = 2;
        let mut rng = nanorand::WyRand::new();
        if (self.wind_update_counter % wind_update_freq == 0)
        {
            let mut wind_offsets : Vec<[f32; 2]> = Vec::new();
            for _ in 0..GRASS_COUNT {
                wind_offsets.push([
                    (self.now.elapsed().as_secs_f32() * 100.0).to_radians().sin().abs() * rng.generate::<f32>(), 
                    (rng.generate::<f32>()),
                ]);
            }
            queue.write_buffer(&self.grass_wind_uniform_buf, 0, bytemuck::cast_slice(&wind_offsets));
        }
        self.wind_update_counter += 1;


        if self.lights_are_dirty {
            self.lights_are_dirty = false;
            for (i, light) in self.lights.iter().enumerate() {
                queue.write_buffer(
                    &self.light_storage_buf,
                    (i * size_of::<LightRaw>()) as wgpu::BufferAddress,
                    bytemuck::bytes_of(&light.to_raw()),
                );
            }
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let vbo_size = self.compute_pass.vertex_size as u64 * Self::VERTS_PER_GRASSBLADE as u64;

        // copy entities VBOs into compute storage
        {
            for i in 0..GRASS_COUNT {

                encoder.copy_buffer_to_buffer(
                    &self.entities[1].const_vertex_buf,
                    0,
                    &self.compute_pass.storage_buf,
                    (i as u64 * vbo_size) as wgpu::BufferAddress,
                    vbo_size
                );
            }
        }
        encoder.push_debug_group("compute passes");
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            // run compute to offset vertices
            cpass.set_pipeline(&self.compute_pass.pipeline);
            cpass.set_bind_group(0, &self.compute_pass.bind_group, &[]);
            cpass.set_bind_group(1, &self.wind_bind_group, &[]);
            cpass.insert_debug_marker("compute grass beziers");
            cpass.dispatch_workgroups(GRASS_COUNT, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        encoder.pop_debug_group();
        // copy into the runtime vertex buffer
        {
            for (i, entity) in self.entities.iter().enumerate() {
                if (i == 0) // jank to ignore the plane
                {
                    continue;
                }

                encoder.copy_buffer_to_buffer(
                    &self.compute_pass.storage_buf,
                    ((i - 1) as u64 * vbo_size) as wgpu::BufferAddress, // jank to ignore the plane
                    &entity.vertex_buf,
                    0,
                    vbo_size
                );
            }
        }


        encoder.push_debug_group("shadow passes");
        for (i, light) in self.lights.iter().enumerate() {
            encoder.push_debug_group(&format!(
                "shadow pass {} (light at position {:?})",
                i, light.pos
            ));

            // The light uniform buffer already has the projection,
            // let's just copy it over to the shadow uniform buffer.
            encoder.copy_buffer_to_buffer(
                &self.light_storage_buf,
                (i * size_of::<LightRaw>()) as wgpu::BufferAddress,
                &self.shadow_pass.uniform_buf,
                0,
                64,
            );

            encoder.insert_debug_marker("render entities");
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &light.target_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.shadow_pass.pipeline);
                pass.set_bind_group(0, &self.shadow_pass.bind_group, &[]);

                let entity = &self.entities[1];
                //for entity in &self.entities {
                    pass.set_bind_group(1, &self.entity_bind_group, &[entity.uniform_offset]);
                    pass.set_index_buffer(entity.index_buf.slice(..), entity.index_format);
                    pass.set_vertex_buffer(0, entity.const_vertex_buf.slice(..));
                    //pass.draw_indexed(0..entity.index_count as u32, 0, 0..1);
                //}
                pass.multi_draw_indexed_indirect(&self.indirect_buffer, 0 as BufferAddress, 1u32);

            }

            encoder.pop_debug_group();
        }
        encoder.pop_debug_group();

        // forward pass
        encoder.push_debug_group("forward rendering pass");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.forward_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.forward_pass.pipeline);
            pass.set_bind_group(0, &self.forward_pass.bind_group, &[]);

            // plane
            {
                let entity = &self.entities[0];
                pass.set_bind_group(1, &self.entity_bind_group, &[entity.uniform_offset]);
                pass.set_index_buffer(entity.index_buf.slice(..), entity.index_format);
                pass.set_vertex_buffer(0, entity.vertex_buf.slice(..));
                pass.draw_indexed(0..entity.index_count as u32, 0, 0..1);
            }

            // grass blades
            let entity = &self.entities[1];
            pass.set_bind_group(1, &self.entity_bind_group, &[entity.uniform_offset]);
            pass.set_index_buffer(entity.index_buf.slice(..), entity.index_format);
            pass.set_vertex_buffer(0, entity.const_vertex_buf.slice(..));
            //pass.draw_indexed(0..entity.index_count as u32, 0, 0..1);
            pass.multi_draw_indexed_indirect(&self.indirect_buffer, 0 as BufferAddress, 1u32);
        }
        encoder.pop_debug_group();

        queue.submit(iter::once(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("shadow");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "shadow",
    image_path: "/examples/src/shadow/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default()
        .downlevel_flags(wgpu::DownlevelFlags::COMPARISON_SAMPLERS)
        // rpi4 on VK doesn't work: https://gitlab.freedesktop.org/mesa/mesa/-/issues/3916
        .expect_fail(wgpu_test::FailureCase::backend_adapter(
            wgpu::Backends::VULKAN,
            "V3D",
        )),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
