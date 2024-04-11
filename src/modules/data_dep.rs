use wgpu::util::DeviceExt;

use crate::setup::WState;

pub async fn data_dep_gpu(state: &mut WState) {
    let cs_module = state
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("data_dep.wgsl").into()),
        });

    let in_sq = vec![0i32; 4];

    let in_bytes: &[u8] = bytemuck::cast_slice(&in_sq);

    let in_buffer = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: in_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
    let out_buf = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: in_bytes.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout =
        state
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind Group Layout"),
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
        state
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
    let pipeline = state
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: "main",
        });

    for i in 0..4 as u32 {
        let start = std::time::Instant::now();
        let pass_bytes = bytemuck::bytes_of(&i);
        let pass_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pass Buffer"),
                contents: pass_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pass_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = state.device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(in_sq.len() as u32, 1, 1);
        }
        state.queue.submit(Some(encoder.finish()));
        println!("Elapsed: {:?}", start.elapsed());
    }

    let mut encoder = state.device.create_command_encoder(&Default::default());
    // Get data out of device
    encoder.copy_buffer_to_buffer(&in_buffer, 0, &out_buf, 0, in_bytes.len() as u64);
    state.queue.submit(Some(encoder.finish()));

    let out_slice = out_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    out_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // More queries
    state.device.poll(wgpu::Maintain::Wait);

    println!("post-poll {:?}", std::time::Instant::now());
    if let Some(Ok(())) = receiver.receive().await {
        let data_raw = &*out_slice.get_mapped_range();
        let data: &[i32] = bytemuck::cast_slice(data_raw);
        println!("{:?}", data);
    }
    // println!("Elapsed: {:?}", start.elapsed());
}
