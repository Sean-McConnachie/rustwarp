use bytemuck::{Pod, Zeroable};
use pollster::FutureExt;
use rand::{distributions::Standard, prelude::*};
use wgpu::util::DeviceExt;

use crate::setup::*;

pub mod impl_prelude {
    pub use super::{WTestable, WType};
    pub use crate::wtest;
    pub use rand::distributions::Distribution as WDistribution;
    pub use rand::distributions::Standard as WStandard;
}

#[derive(Debug, PartialEq)]
pub enum WType {
    Primitive(&'static str),
    Struct(&'static str),
}

pub trait WTestable {
    fn wgsl_type() -> WType;

    fn inner_type() -> &'static str {
        match Self::wgsl_type() {
            WType::Primitive(inner) => inner,
            WType::Struct(inner) => inner,
        }
    }
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
    let input_bytes = wbyte_cast!(&input_values);

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

        let cs_module = wgpu_shader_load!("Compute shader", state.device, shader);

        let input_buf = wgpu_buf_init!(
            "Input buffer",
            state.device,
            input_bytes,
            [STORAGE | COPY_SRC]
        );
        let copied_buf = wgpu_buf!(
            "Copied buffer",
            state.device,
            input_bytes.len() as u64,
            [STORAGE | COPY_DST | COPY_SRC],
            false
        );
        let output_buf = wgpu_buf!(
            "Output buffer",
            state.device,
            input_bytes.len() as u64,
            [MAP_READ | COPY_DST],
            false
        );

        let bind_group_layout =
            wgpu_bind_group_layout_compute!(state.device, [(0, true), (1, false)]);

        let compute_pipeline_layout =
            wgpu_compute_pipeline_layout!(state.device, &[&bind_group_layout]);

        let pipeline =
            wgpu_compute_pipeline!(state.device, &compute_pipeline_layout, &cs_module, "main");

        let bind_group = wgpu_bind_group!(
            state.device,
            &bind_group_layout,
            [
                (0, input_buf.as_entire_binding()),
                (1, copied_buf.as_entire_binding())
            ]
        );

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
