use crate::{
    core::WMat3x3Affine,
    image::{Image, Pix, Size},
    setup::WState,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::tester::impl_prelude::*;

#[derive(Copy, Clone, Debug, PartialEq)]
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

pub struct WarpPerspectiveGpu {
    pub state: WState,
    cs_module: wgpu::ShaderModule,
    pub transform: Vec<ImageTransform>,
    pub interp: Interpolation,

    pub src_bytes: Vec<u8>,
    pub transform_bytes: Vec<u8>,

    pub src: Vec<Image>,
    pub dst: Vec<Image>,
}

impl WarpPerspectiveGpu {
    pub fn new(
        state: WState,
        transform: Vec<ImageTransform>,
        interp: Interpolation,
        src: Vec<Image>,
        dst: Vec<Image>,
    ) -> Self {
        let cs_module = state
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Image transform shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("multiple_warp.wgsl").into()),
            });

        // Make relevant bytes
        let src_bytes = src
            .iter()
            .map(|im| bytemuck::cast_slice(im.data.as_ref()))
            .flatten()
            .copied()
            .collect::<Vec<u8>>();

        let transform_bytes = transform
            .iter()
            .map(|t| bytemuck::bytes_of(t))
            .flatten()
            .copied()
            .collect::<Vec<u8>>();

        Self {
            state,
            cs_module,
            transform,
            interp,
            src_bytes,
            transform_bytes,
            src,
            dst,
        }
    }

    pub async fn render_pass(&mut self) {
        warp_perspective_gpu(
            &mut self.state,
            &self.cs_module,
            &self.transform,
            self.interp,
            &mut self.src_bytes,
            &self.transform_bytes,
            &mut self.dst,
        )
        .await;
    }
}

pub async fn warp_perspective_gpu(
    state: &mut WState,
    cs_module: &wgpu::ShaderModule,
    _transform: &Vec<ImageTransform>,
    interp: Interpolation,
    src_bytes: &mut Vec<u8>,
    transform_bytes: &Vec<u8>,
    // src: &Vec<Image>,
    dst: &mut Vec<Image>,
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

    // Make relevant buffers
    let transform_buf = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform buffer"),
            contents: &transform_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
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

    let mut i = 0;
    loop {
        i += 1;
        src_bytes[i] = 255;

        let start = std::time::Instant::now();
        let src_buf = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input image buffer"),
                contents: &src_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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

        // Encoder
        let mut encoder = state.device.create_command_encoder(&Default::default());
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 0);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(dst[0].size.x as u32, dst[0].size.y as u32, dst.len() as u32);
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
            let chunk_size = dst[0].size.x * dst[0].size.y;
            for (dst, data) in dst.iter_mut().zip(data.chunks(chunk_size)) {
                dst.data.copy_from_slice(data);
            }
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

        out_buf.unmap();
        query_buf.unmap();

        for (i, d) in dst.iter().enumerate() {
            d.rgb_image()
                .unwrap()
                .save(format!("outputs/output_{}.png", i))
                .unwrap();
        }
    }
}
