use std::fmt::Display;

use bytemuck::{Pod, Zeroable};
use pollster::FutureExt;
use rustwarp::{
    device::{core::WMat, setup::WState},
    *,
};
use wgpu::util::DeviceExt;

wvec_impls!(
    struct Pix {
        colour: wvec3!(u32, 4),
    }
);

impl Display for Pix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {})",
            self.colour.x, self.colour.y, self.colour.z
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Point {
    x: usize,
    y: usize,
}

impl Point {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

type Pos = Point;
type Size = Point;

struct Image {
    data: Vec<Pix>,
    size: Size,
}

impl Image {
    pub fn new(size: Size) -> Self {
        Self {
            data: vec![Pix::default(); size.x * size.y],
            size,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &Pix {
        &self.data[y * self.size.x as usize + x]
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut Pix {
        &mut self.data[y * self.size.x as usize + x]
    }

    pub fn rgb_image(&self) -> Result<image::RgbImage, Box<dyn std::error::Error>> {
        TryInto::<image::RgbImage>::try_into(self)
    }
}

impl TryInto<image::RgbImage> for &Image {
    type Error = Box<dyn std::error::Error>;
    fn try_into(self) -> Result<image::RgbImage, Self::Error> {
        let mut im = image::RgbImage::new(self.size.x as u32, self.size.y as u32);
        for (x, y, pixel) in im.enumerate_pixels_mut() {
            let p = self.get(x as usize, y as usize);
            *pixel = image::Rgb([
                p.colour.x.try_into()?,
                p.colour.y.try_into()?,
                p.colour.z.try_into()?,
            ]);
        }
        Ok(im)
    }
}

wvec_impls!(
    struct ImageTransform {
        dimensions: wvec2!(u32, 8),
        matrix: WM,
    }
);

impl ImageTransform {
    pub fn new(dimensions: Size, matrix: WM) -> Self {
        let mut s = Self::default();
        s.dimensions.x = dimensions.x as u32;
        s.dimensions.y = dimensions.y as u32;
        s.matrix = matrix;
        s
    }
}

async fn warp_perspective_gpu(
    state: &mut WState,
    transform: &ImageTransform,
    src: &Image,
    dst: &mut Image,
) {
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
            source: wgpu::ShaderSource::Wgsl(
                include_str!("device/perspective_transform.wgsl").into(),
            ),
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
            entry_point: "main",
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

fn warp_perspective_cpu(transform: &ImageTransform, src: &Image, dst: &mut Image) {
    for y in 0..dst.size.y {
        for x in 0..dst.size.x {
            let pos = (x, y, 1.0);
            let &m = &transform.matrix.matrix();
            // remember, row major
            let from_pos = [
                m[0][0] * pos.0 as f32 + m[1][0] * pos.1 as f32 + m[2][0] * pos.2,
                m[0][1] * pos.0 as f32 + m[1][1] * pos.1 as f32 + m[2][1] * pos.2,
                m[0][2] * pos.0 as f32 + m[1][2] * pos.1 as f32 + m[2][2] * pos.2,
            ];
            let from_pos = (
                from_pos[0] / from_pos[2],
                from_pos[1] / from_pos[2],
                from_pos[2],
            );
            let from_pos = (from_pos.0 as usize, from_pos.1 as usize, from_pos.2);
            if from_pos.0 < src.size.x && from_pos.1 < src.size.y {
                let pix = src.get(from_pos.0, from_pos.1);
                dst.get_mut(x, y).colour = pix.colour;
            }
        }
    }
}

type M = [[f32; 3]; 3];
type WM = WMat<f32, 3, 3, 4>;
// type WM = [[f32; 4]; 4];
// const MAT: M = [
//     [0.7071067811865479, 0.7071067811865475, -165.47831816805157],
//     [-0.7071067811865475, 0.7071067811865476, 370.2106781186547],
//     [3.552713678800502e-19, -3.4045953094161123e-35, 1.0],
// ];
const IDENTITY: M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const SCALE_XY: M = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const TRANSLATE_XY: M = [[1.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

const ROTATE: M = [
    [0.9594929736144975, 0.28173255684142967, 0.0],
    [-0.2817325568414297, 0.9594929736144975, 0.0],
    [0.0, 0.0, 1.0],
];

const MAT: M = ROTATE;

async fn test_copy_and_back(
    state: &mut WState,
    transform: &ImageTransform,
    src: &Image,
    dst: &mut Image,
) {
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
            source: wgpu::ShaderSource::Wgsl(include_str!("device/test_copy_and_back.wgsl").into()),
        });

    // Make relevant bytes
    let src_bytes: &[u8] = bytemuck::cast_slice(&src.data);
    let transform_bytes = bytemuck::bytes_of(transform);
    println!("{:?}", transform_bytes);

    // Make relevant buffers
    let src_buf = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input image buffer"),
            contents: src_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
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
            entry_point: "main",
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
        // cpass.dispatch_workgroups(dst.size.x as u32, dst.size.y as u32, 1);
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

    let print_vec = |v| {
        for i in v {
            print!("{}, ", i);
        }
    };

    print_vec(&src.data);
    println!();
    println!();
    print_vec(&dst.data);
    println!();

    for y in 0..dst.size.y {
        for x in 0..dst.size.x {
            let p1 = src.get(x, y);
            let p2 = dst.get(x, y);
            if p1.colour.x != p2.colour.x
                || p1.colour.y != p2.colour.y
                || p1.colour.z != p2.colour.z
            {
                println!("Mismatch at ({}, {})", x, y);
                println!("{:?}", p1);
                println!("{:?}", p2);
                panic!();
            }
        }
    }
}

fn main() {
    let mut state = WState::new().block_on();
    let size = Size::new(800, 800);
    let input = {
        let mut im = Image::new(size);
        const BOX_SIZE: usize = 200 / 2;
        let hw = size.x / 2;
        let hh = size.y / 2;
        let total = BOX_SIZE * 8;
        let mut c = 0;
        let mut set_pix = |pix: &mut Pix| {
            pix.colour.x = 255;
            pix.colour.y = 255;
            let z = ((c as f32 / total as f32) * 255.0) as u32;
            let z = z.min(255);
            pix.colour.z = z;
            c += 1;
        };
        for y in hh - BOX_SIZE..hh + BOX_SIZE {
            set_pix(im.get_mut(hw - BOX_SIZE, y));
            set_pix(im.get_mut(hw + BOX_SIZE, y));
        }
        for x in hw - BOX_SIZE..hw + BOX_SIZE {
            set_pix(im.get_mut(x, hh - BOX_SIZE));
            set_pix(im.get_mut(x, hh + BOX_SIZE));
        }
        im
    };

    let _ = input
        .rgb_image()
        .unwrap()
        .save("outputs/input.png")
        .unwrap();

    let transform = {
        let mut mat = WM::default();
        for x in 0..3 {
            for y in 0..3 {
                mat.set(x, y, MAT[y][x]);
            }
        }
        ImageTransform::new(size, mat)
    };

    if false {
        let mut input = Image::new(size);
        for y in 0..size.y {
            for x in 0..size.x {
                let pix = input.get_mut(x, y);
                pix.colour.x = (x as f32 / size.x as f32 * 255.0) as u32;
                pix.colour.y = (y as f32 / size.y as f32 * 255.0) as u32;
                pix.colour.z = 0;
            }
        }

        let mut cpu_output = Image::new(size);
        test_copy_and_back(&mut state, &transform, &input, &mut cpu_output).block_on();
        return;
    }

    let mut cpu_output = Image::new(size);
    warp_perspective_cpu(&transform, &input, &mut cpu_output);
    let _ = cpu_output
        .rgb_image()
        .unwrap()
        .save("outputs/cpuoutput.png")
        .unwrap();

    let mut gpu_output = Image::new(size);
    warp_perspective_gpu(&mut state, &transform, &input, &mut gpu_output).block_on();
    let _ = gpu_output
        .rgb_image()
        .unwrap()
        .save("outputs/gpuoutput.png")
        .unwrap();
}
