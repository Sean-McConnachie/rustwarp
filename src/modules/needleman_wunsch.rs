use bytemuck::{Pod, Zeroable};

use crate::{setup::WState, tester::impl_prelude::*};

type NucleotideInt = u32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Nucleotide {
    A,
    C,
    G,
    T,
}

impl From<&str> for Nucleotide {
    fn from(value: &str) -> Self {
        match value.into() {
            "a" | "A" => Nucleotide::A,
            "c" | "C" => Nucleotide::C,
            "g" | "G" => Nucleotide::G,
            "t" | "T" => Nucleotide::T,
            _ => panic!("Invalid nucleotide"),
        }
    }
}

impl Into<NucleotideInt> for &Nucleotide {
    fn into(self) -> NucleotideInt {
        match self {
            Nucleotide::A => 0,
            Nucleotide::C => 1,
            Nucleotide::G => 2,
            Nucleotide::T => 3,
        }
    }
}

impl From<NucleotideInt> for Nucleotide {
    fn from(value: NucleotideInt) -> Self {
        match value {
            0 => Nucleotide::A,
            1 => Nucleotide::C,
            2 => Nucleotide::G,
            3 => Nucleotide::T,
            _ => panic!("Invalid nucleotide"),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence(pub Vec<Nucleotide>);

impl From<&str> for Sequence {
    fn from(value: &str) -> Self {
        Sequence(
            value
                .chars()
                .map(|c| Nucleotide::from(c.to_string().to_lowercase().as_str()))
                .collect(),
        )
    }
}

macro_rules! max {
    ($x:expr) => ($x);
    ($x:expr, $($y:expr),+) => (::std::cmp::max($x, max!($($y),+)));
}

pub fn needleman_wunsch_cpu(
    seq1: &Sequence,
    seq2: &Sequence,
    gap_penalty: i64,
    mismatch_penalty: i64,
    match_score: i64,
) -> i64 {
    let s1 = &seq1.0;
    let s2 = &seq2.0;

    let mut matrix = vec![vec![0; s2.len() + 1]; s1.len() + 1];

    for i in 0..s1.len() + 1 {
        matrix[i][0] = i as i64 * gap_penalty;
    }

    for j in 0..s2.len() + 1 {
        matrix[0][j] = j as i64 * gap_penalty;
    }

    for i in 1..s1.len() + 1 {
        for j in 1..s2.len() + 1 {
            let score = if s1[i - 1] == s2[j - 1] {
                match_score
            } else {
                mismatch_penalty
            };

            matrix[i][j] = max!(
                matrix[i - 1][j - 1] + score,
                matrix[i - 1][j] + gap_penalty,
                matrix[i][j - 1] + gap_penalty
            );
        }
    }

    matrix[s1.len()][s2.len()]
}

pub use gpu::*;
pub mod gpu {
    use wgpu::util::DeviceExt;

    use super::*;

    pub type Score = i32;
    const UNDEFINED: Score = -2147483648;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Pod, Zeroable, Default, PartialEq)]
    struct NWInfo {
        seq1_len: u32,
        seq2_len: u32,
        gap_penalty: i32,
        mismatch_penalty: i32,
        match_score: i32,
    }

    impl WTestable for NWInfo {
        fn wgsl_type() -> WType {
            WType::Struct("seq1_len: u32, seq2_len: u32, gap_penalty: i32, mismatch_penalty: i32, match_score: i32")
        }
    }

    impl WDistribution<NWInfo> for WStandard {
        fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> NWInfo {
            NWInfo {
                seq1_len: rng.gen(),
                seq2_len: rng.gen(),
                gap_penalty: rng.gen(),
                mismatch_penalty: rng.gen(),
                match_score: rng.gen(),
                ..Default::default()
            }
        }
    }
    wtest!(NWInfo, 1021);

    pub async fn needleman_wunsch_gpu(
        state: &mut WState,
        seq1: &Sequence,
        seq2: &Sequence,
        gap_penalty: i32,
        mismatch_penalty: i32,
        match_score: i32,
    ) -> i64 {
        panic!("Dynamic programming requires a more complex implementation.");
        let cs_module = state
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Needleman Wunsch Compute Shader Module"),
                source: wgpu::ShaderSource::Wgsl(include_str!("needleman_wunsch.wgsl").into()),
            });

        let start = std::time::Instant::now();
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

        let info = NWInfo {
            seq1_len: seq1.0.len() as u32 + 1,
            seq2_len: seq2.0.len() as u32 + 1,
            gap_penalty: gap_penalty as i32,
            mismatch_penalty: mismatch_penalty as i32,
            match_score: match_score as i32,
        };

        let matrix = {
            let mut m = vec![UNDEFINED; (seq2.0.len() + 1) * (seq1.0.len() + 1)];
            for i in 0..seq1.0.len() + 1 {
                m[i * (seq2.0.len() + 1)] = i as i32 * gap_penalty;
            }
            for j in 0..seq2.0.len() + 1 {
                m[j] = j as i32 * gap_penalty;
            }
            m
        };

        let seq1_ints: Vec<NucleotideInt> = seq1.0.iter().map(|n| n.into()).collect();
        let seq2_ints: Vec<NucleotideInt> = seq2.0.iter().map(|n| n.into()).collect();

        let info_bytes = bytemuck::bytes_of(&info);
        let seq1_bytes: &[u8] = bytemuck::cast_slice(&seq1_ints);
        let seq2_bytes: &[u8] = bytemuck::cast_slice(&seq2_ints);
        let matrix_bytes: &[u8] = bytemuck::cast_slice(&matrix);

        let info_buf = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NW Info Buffer"),
                contents: info_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let seq1_buf = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sequence 1 Buffer"),
                contents: seq1_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let seq2_buf = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sequence 2 Buffer"),
                contents: seq2_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let matrix_buf = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix Buffer"),
                contents: matrix_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let out_buf = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix output buffer"),
            size: matrix_bytes.len() as u64,
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

        let bind_group_layout =
            state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("NW Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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

        let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NW Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seq1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: seq2_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: matrix_buf.as_entire_binding(),
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
            cpass.dispatch_workgroups(seq2.0.len() as u32 + 1, seq1.0.len() as u32 + 1, 1);
        }
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 1);
        }

        // Get data out of device
        encoder.copy_buffer_to_buffer(&matrix_buf, 0, &out_buf, 0, matrix_bytes.len() as u64);
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
        let mut score = UNDEFINED;
        if let Some(Ok(())) = receiver.receive().await {
            let data_raw = &*out_slice.get_mapped_range();
            let data: &[Score] = bytemuck::cast_slice(data_raw);
            println!("{:?}", data);
            score = data[data.len() - 1];
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

        score as i64
    }
}

mod tests {
    use super::*;

    #[allow(dead_code)]
    struct NeedlemanWunschData {
        seq1: Sequence,
        seq2: Sequence,
        gap_penalty: i64,
        mismatch_penalty: i64,
        match_score: i64,
        result: i64,
    }

    #[test]
    fn test_needleman_wunsch_cpu() {
        let tests: &[NeedlemanWunschData] = &[
            NeedlemanWunschData {
                seq1: "GCATGCG".into(),
                seq2: "GATTACA".into(),
                match_score: 1,
                mismatch_penalty: -1,
                gap_penalty: -1,
                result: 0,
            },
            NeedlemanWunschData {
                seq1: "ACTGATTCA".into(),
                seq2: "ACGCATCA".into(),
                match_score: 2,
                mismatch_penalty: -2,
                gap_penalty: -2,
                result: 8,
            },
        ];

        for test in tests {
            let score = needleman_wunsch_cpu(
                &test.seq1,
                &test.seq2,
                test.gap_penalty,
                test.mismatch_penalty,
                test.match_score,
            );

            assert_eq!(score, test.result);
        }
    }
}
