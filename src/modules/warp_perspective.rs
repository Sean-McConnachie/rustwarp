use crate::{
    image::{Image, Pix, Size},
    setup::WState,
    types::WMat3x3Affine,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::tester::impl_prelude::*;

pub enum Interpolation {
    None,
    Bilinear,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod, PartialEq)]
pub struct ImageTransform {
    pub dimensions: wvec2!(u32, 8),
    pub inverse_matrix: WMat3x3Affine,
}

impl ImageTransform {
    pub fn new(dimensions: Size, matrix: WMat3x3Affine) -> Self {
        let mut s = Self::default();
        s.dimensions.x = dimensions.x as u32;
        s.dimensions.y = dimensions.y as u32;
        s.inverse_matrix = matrix;
        s
    }
}

impl WTestable for ImageTransform {
    fn wgsl_type() -> WType {
        WType::Struct("a: vec2<u32>, b: mat3x3<f32>")
    }
}

impl WDistribution<ImageTransform> for WStandard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> ImageTransform {
        ImageTransform {
            dimensions: rng.gen(),
            inverse_matrix: rng.gen(),
        }
    }
}

wtest!(ImageTransform, 256);

pub fn warp_perspective_cpu(
    transform: &ImageTransform,
    interp: Interpolation,
    src: &Image,
    dst: &mut Image,
) {
    for y in 0..dst.size.y {
        for x in 0..dst.size.x {
            let pi = (x as f32, y as f32, 1.0);
            let m = &transform.inverse_matrix.matrix();
            let from_pos = [
                m[0][0] * pi.0 + m[1][0] * pi.1 + m[2][0] * pi.2,
                m[0][1] * pi.0 + m[1][1] * pi.1 + m[2][1] * pi.2,
                m[0][2] * pi.0 + m[1][2] * pi.1 + m[2][2] * pi.2,
            ];
            let pt = (from_pos[0] / from_pos[2], from_pos[1] / from_pos[2]);
            let pn = (pt.0 as usize, pt.1 as usize);
            match interp {
                Interpolation::None => {
                    if pn.0 < src.size.x && pn.1 < src.size.y {
                        let pix = src.get(pn.0, pn.1);
                        *dst.get_mut(x, y) = *pix;
                    } else {
                        *dst.get_mut(x, y) = Pix::default();
                    }
                }
                Interpolation::Bilinear => {
                    if pn.0 <= 0 || pn.0 >= dst.size.x - 1 || pn.1 <= 0 || pn.1 >= dst.size.y - 1 {
                        *dst.get_mut(x, y) = Pix::new(0, 0, 0, 0);
                        continue;
                    }
                    let pd = (pt.0 - pn.0 as f32, pt.1 - pn.1 as f32);
                    let v0 = src
                        .get(pn.0, pn.1)
                        .mult((1 as f32 - pd.0) * (1.0f32 - pd.1));
                    let v1 = src.get(pn.0 + 1, pn.1).mult(pd.0 * (1 as f32 - pd.1));
                    let v2 = src.get(pn.0, pn.1 + 1).mult((1 as f32 - pd.0) * pd.1);
                    let v3 = src.get(pn.0 + 1, pn.1 + 1).mult(pd.0 * pd.1);
                    *dst.get_mut(x, y) = v0.add(&v1).add(&v2).add(&v3);
                }
            }
        }
    }
}

pub async fn warp_perspective_gpu(
    state: &mut WState,
    transform: &ImageTransform,
    interp: Interpolation,
    src: &Image,
    dst: &mut Image,
) {
    let entry_point = match interp {
        Interpolation::None => "interpolation_none",
        Interpolation::Bilinear => "interpolation_bilinear",
    };

    let features = state.device.features();
    let query_set = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        Some(state.device.create_query_set(&wgpu::QuerySetDescriptor {
            count: 2,
            ty: wgpu::QueryType::Timestamp,
            label: None,
        }))
    } else {
        None
    };

    let cs_module = state
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image transform shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("warp_perspective.wgsl").into()),
        });

    // Make relevant bytes
    let src_bytes: &[u8] = bytemuck::cast_slice(&src.data);
    let transform_bytes = bytemuck::bytes_of(transform);

    // Make relevant buffers
    let src_buf = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input image buffer"),
            contents: src_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let transform_buf = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform buffer"),
            contents: transform_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    assert_eq!(src.size, dst.size);
    let dst_buf = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output image buffer"),
        size: src_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output image buffer"),
        size: src_bytes.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let query_buf = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query buffer"),
            contents: &[0; 16],
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });

    // Bind group layout
    let bind_group_layout =
        state
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            // TODO: Change to uniform?
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

    // Pipeline
    let compute_pipeline_layout =
        state
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Computer Pipeline"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
    let pipeline = state
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Main pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point,
        });

    // Bind group
    let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dst_buf.as_entire_binding(),
            },
        ],
    });

    let start = std::time::Instant::now();

    // Encoder
    let mut encoder = state.device.create_command_encoder(&Default::default());
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(dst.size.x as u32, dst.size.y as u32, 1);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }

    // Get data out of device
    encoder.copy_buffer_to_buffer(&dst_buf, 0, &out_buf, 0, src_bytes.len() as u64);
    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    state.queue.submit(Some(encoder.finish()));

    let out_slice = out_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    out_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // More queries
    let query_slice = query_buf.slice(..);
    let _query_future = query_slice.map_async(wgpu::MapMode::Read, |_| ());
    println!("pre-poll {:?}", std::time::Instant::now());
    state.device.poll(wgpu::Maintain::Wait);

    println!("post-poll {:?}", std::time::Instant::now());
    if let Some(Ok(())) = receiver.receive().await {
        let data_raw = &*out_slice.get_mapped_range();
        let data: &[Pix] = bytemuck::cast_slice(data_raw);
        println!("{:?}", data.len());
        dst.data = data.into();
    }
    if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        let ts_period = state.queue.get_timestamp_period();
        let ts_data_raw = &*query_slice.get_mapped_range();
        let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
        println!(
            "compute shader elapsed: {:?}ms",
            (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
        );
    }

    println!("Elapsed: {:?}", start.elapsed());
}
