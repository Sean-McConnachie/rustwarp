use bytemuck::{Pod, Zeroable};
use pollster::FutureExt;
use rand::{distributions::Standard, prelude::*};
use wgpu::util::DeviceExt;

use crate::setup::WState;

pub mod impl_prelude {
    pub use super::{WTestable, WType};
    pub use crate::wtest;
    pub use rand::distributions::Distribution as WDistribution;
    pub use rand::distributions::Standard as WStandard;
}

pub enum WType {
    Primitive(&'static str),
    Struct(&'static str),
}

pub trait WTestable {
    fn wgsl_type() -> WType;
}

impl WTestable for bool {
    fn wgsl_type() -> WType {
        WType::Primitive("bool")
    }
}

impl WTestable for u32 {
    fn wgsl_type() -> WType {
        WType::Primitive("u32")
    }
}

impl WTestable for i32 {
    fn wgsl_type() -> WType {
        WType::Primitive("i32")
    }
}

impl WTestable for f32 {
    fn wgsl_type() -> WType {
        WType::Primitive("f32")
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub struct WTestFail<T> {
    pub failed_at_value: usize,
    pub total_values: usize,
    pub expected: Vec<T>,
    pub got: Vec<T>,
}

#[derive(Debug, PartialEq)]
pub enum WTestResult<T> {
    Success,
    TestError(String),
    TestFail(WTestFail<T>),
}

pub fn perform_test<T>(n: usize) -> WTestResult<T>
where
    T: WTestable + Zeroable + Pod + PartialEq + Copy,
    Standard: Distribution<T>,
{
    let mut rng = thread_rng();
    let input_values: Vec<T> = (0..n).map(|_| rng.gen()).collect();
    let input_bytes = bytemuck::cast_slice(&input_values);

    const SHADER: &'static str = r#"
        {struct_wgsl_type}
    
        @group(0)
        @binding(0)
        var<storage, read> input: array<{wgsl_type}>;

        @group(0)
        @binding(1)
        var<storage, read_write> output: array<{wgsl_type}>;

        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            output[index] = input[index];
        }
    "#;
    let shader = match T::wgsl_type() {
        WType::Primitive(t) => SHADER
            .replace("{struct_wgsl_type}", "")
            .replace("{wgsl_type}", t),
        WType::Struct(t) => SHADER
            .replace(
                "{struct_wgsl_type}",
                format!("struct StructType {{ {} }};", t).as_str(),
            )
            .replace("{wgsl_type}", "StructType"),
    };

    // async anoynmous block
    let output_result = async {
        let state = WState::new().await;

        let cs_module = state
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute shader"),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });

        let input_buf = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input image buffer"),
                contents: input_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let copied_buf = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Copied buffer"),
            size: input_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output buffer"),
            size: input_bytes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bind group layout"),
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
                    label: Some("Compute pipeline layout"),
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
        let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: copied_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = state.device.create_command_encoder(&Default::default());

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(n as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&copied_buf, 0, &output_buf, 0, input_bytes.len() as u64);
        state.queue.submit(Some(encoder.finish()));

        let output_slice = output_buf.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        output_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        state.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let data_raw = &*output_slice.get_mapped_range();
            let data: &[T] = bytemuck::cast_slice(data_raw);
            return Ok(data.to_vec());
        }
        Err(())
    }
    .block_on();

    let output_values = match output_result {
        Ok(v) => v,
        Err(_) => {
            return WTestResult::TestError("Failed execute test!".to_string());
        }
    };

    for (i, (expected, got)) in input_values.iter().zip(output_values.iter()).enumerate() {
        if expected != got {
            return WTestResult::TestFail(WTestFail {
                failed_at_value: i,
                total_values: n,
                expected: input_values,
                got: output_values,
            });
        }
    }
    WTestResult::Success
}

mod tests {
    #[macro_export]
    macro_rules! wtest {
        ($typ:ty, $n:expr) => {
            paste::paste! {
                #[test]
                #[allow(non_snake_case)]
                fn [<wtest_ $typ _ $n>]() {
                    assert_eq!(
                        crate::tester::perform_test::<$typ>($n),
                        crate::tester::WTestResult::Success
                    );
                }
            }
        };
    }
}
