// device, queue, config, render_pipeline, vertex_buffer, index_buffer, diffuse_bind_group
//
// adapter, features, device, queue, shader_module, input_buffer, out_buffer, query_buffer, bind_group_layout
// computer_pipeline_layout, pipeline, bind_group, encoder, oneshot_chan,

use crate::host;
use wgpu::util::DeviceExt;

type Vec2U32 = [u32; 2];
type Vec4U32 = [u32; 4];

type Vec2F32 = [f32; 2];
type Vec3F32 = [f32; 3];
type Vec4F32 = [f32; 4];

type Mat3x3F32 = [Vec4F32; 3];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ImTranform {
    odim: Vec4U32,
    mat: Mat3x3F32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, Default)]
struct Pixel {
    // color: Vec3F32,
    x: f32,
    y: f32,
    z: f32,
    _pad: [f32; 1],
}

pub struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,

    calc_bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    input_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    transform: Mat3x3F32,
}

impl State {
    pub async fn new() -> () {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                    limits: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
        let features = device.features();
        let query_set = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                count: 2,
                ty: wgpu::QueryType::Timestamp,
                label: None,
            }))
        } else {
            None
        };

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image transform shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("transform.wgsl").into()),
        });

        let in_transform = ImTranform {
            odim: [2, 3, 0, 0],
            mat: [[1.1f32; 4], [1.0f32; 4], [1.0f32; 4]],
        };
        let in_transform_bytes: &[u8] = bytemuck::bytes_of(&in_transform);
        dbg!(in_transform_bytes.len());
        let in_transform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input transform buffer"),
            contents: in_transform_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, // | wgpu::BufferUsages::COPY_DST,
        });

        // let in_raw: Vec<[f32; 4]> = vec![[1.0; 4]; 6];
        let in_raw: Vec<Pixel> = vec![
            Pixel {
                x: 1.0,
                y: 1.0,
                z: 1.0,
                ..Default::default()
            };
            6
        ];
        // let in_raw: Vec<f32> = vec![0.0, 1.0];
        let in_bytes: &[u8] = bytemuck::cast_slice(&in_raw[..]);
        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input image buffer"),
            contents: in_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output image buffer"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let query_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query buffer"),
            contents: &[0; 16],
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_transform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 0);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(3, 2, 1);
        }
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 1);
        }
        encoder.copy_buffer_to_buffer(&in_buf, 0, &out_buf, 0, in_bytes.len() as u64);
        if let Some(query_set) = &query_set {
            encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
        }
        queue.submit(Some(encoder.finish()));

        let buf_slice = out_buf.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        let query_slice = query_buf.slice(..);
        // Assume that both buffers become available at the same time. A more careful
        // approach would be to wait for both notifications to be sent.
        let _query_future = query_slice.map_async(wgpu::MapMode::Read, |_| ());
        println!("pre-poll {:?}", std::time::Instant::now());
        device.poll(wgpu::Maintain::Wait);
        println!("post-poll {:?}", std::time::Instant::now());
        if let Some(Ok(())) = receiver.receive().await {
            let data_raw = &*buf_slice.get_mapped_range();
            let data: &[f32] = bytemuck::cast_slice(data_raw);
            println!("data: {:?}", &*data);
        }
        if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            let ts_period = queue.get_timestamp_period();
            let ts_data_raw = &*query_slice.get_mapped_range();
            let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
            println!(
                "compute shader elapsed: {:?}ms",
                (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
            );
        }

        // Self { device, queue }
    }
}
