use bytemuck::{Pod, Zeroable};

use crate::{setup::*, tester::impl_prelude::*};

type NucleotideInt = u32;

const PRINT: bool = true;

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

impl WDistribution<Nucleotide> for WStandard {
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Nucleotide {
        match rng.gen_range(0..4) {
            0 => Nucleotide::A,
            1 => Nucleotide::C,
            2 => Nucleotide::G,
            3 => Nucleotide::T,
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

macro_rules! min {
    ($x:expr) => ($x);
    ($x:expr, $($y:expr),+) => (::std::cmp::min($x, min!($($y),+)));
}

pub fn needleman_wunsch_cpu(
    seq1: &Sequence,
    seq2: &Sequence,
    gap_penalty: i32,
    mismatch_penalty: i32,
    match_score: i32,
) -> i32 {
    let (s1, s2) = if seq1.0.len() >= seq2.0.len() {
        (&seq2.0, &seq1.0)
    } else {
        (&seq1.0, &seq2.0)
    };

    let mut matrix = vec![vec![0; s2.len() + 1]; s1.len() + 1];

    for i in 0..s1.len() + 1 {
        matrix[i][0] = i as i32 * gap_penalty;
    }

    for j in 0..s2.len() + 1 {
        matrix[0][j] = j as i32 * gap_penalty;
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

    if PRINT {
        for i in 0..s1.len() + 1 {
            for j in 0..s2.len() + 1 {
                print!("{:4} ", matrix[i][j]);
            }
            println!();
        }
    }

    matrix[s1.len()][s2.len()]
}

pub use gpu::*;
pub mod gpu {
    use wgpu::util::DeviceExt;

    use super::*;

    pub type Score = i32;

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
    ) -> i32 {
        let (seq1, seq2) = if seq1.0.len() >= seq2.0.len() {
            (seq2, seq1)
        } else {
            (seq1, seq2)
        };
        const WG_COLS: u32 = 2;
        const WG_ROWS: u32 = 2;
        let n = seq2.0.len();
        let m = seq1.0.len();

        let round_up = |numerator: u32, denominator: u32| -> u32 {
            (numerator + denominator - 1) / denominator
        };

        let n_blocks = round_up(1 + n as u32, WG_COLS);
        let m_blocks = round_up(1 + m as u32, WG_ROWS);

        // let cell_counts = {
        //     let mut counts = vec![vec![0; n_blocks as usize]; m_blocks as usize];
        //     for j in 0..m_blocks {
        //         for i in 0..n_blocks {
        //             let x = if i == n_blocks - 1 {
        //                 1 + n as u32 - i * WG_COLS
        //             } else {
        //                 WG_COLS
        //             };
        //             let y = if j == m_blocks - 1 {
        //                 1 + m as u32 - j * WG_ROWS
        //             } else {
        //                 WG_ROWS
        //             };
        //             counts[j as usize][i as usize] = x * y;
        //         }
        //     }
        //     counts
        // };

        assert!(n >= m);
        let info = NWInfo {
            seq1_len: seq1.0.len() as u32,
            seq2_len: seq2.0.len() as u32,
            gap_penalty: gap_penalty as i32,
            mismatch_penalty: mismatch_penalty as i32,
            match_score: match_score as i32,
        };

        let dev = &state.device;
        let shader = wstring_replace!(
            include_str!("needleman_wunsch.wgsl"),
            [
                ("{{WG_COLS}}", &WG_COLS.to_string()),
                ("{{WG_ROWS}}", &WG_ROWS.to_string())
            ]
        );
        let cs_module = wgpu_shader_load!(state.device, shader);
        let mut ts_query = WTsQueryState::new(&state.device, 2);

        let scores: Vec<Score> = {
            let mut s = vec![0; ((n + 1) * (m + 1)) as usize];
            for i in 0..n + 1 {
                s[i] = i as i32 * gap_penalty;
            }
            for j in 0..m + 1 {
                s[j * (n + 1) as usize] = j as i32 * gap_penalty;
            }
            s
        };
        let print_scores = |scores: &[Score], n: usize, m: usize| {
            if PRINT {
                for j in 0..m {
                    for i in 0..n {
                        print!("{:4} ", scores[j * n + i]);
                    }
                    println!();
                }
            }
        };

        let seq1_ints: Vec<NucleotideInt> = seq1.0.iter().map(|n| n.into()).collect();
        let seq2_ints: Vec<NucleotideInt> = seq2.0.iter().map(|n| n.into()).collect();

        let info_bytes = wbyte_of!(&info);
        let seq1_bytes: &[u8] = wbyte_cast!(&seq1_ints);
        let seq2_bytes: &[u8] = wbyte_cast!(&seq2_ints);
        let score_bytes: &[u8] = wbyte_cast!(&scores);

        let info_buf = wgpu_buf_init!(dev, info_bytes, [STORAGE | COPY_SRC]);
        let seq1_buf = wgpu_buf_init!(dev, seq1_bytes, [STORAGE | COPY_SRC]);
        let seq2_buf = wgpu_buf_init!(dev, seq2_bytes, [STORAGE | COPY_SRC]);
        let scores_buf = wgpu_buf_init!(dev, score_bytes, [STORAGE | COPY_SRC | COPY_DST]);
        let out_buf = wgpu_buf!(dev, score_bytes.len() as u64, [MAP_READ | COPY_DST], false);

        let bind_group_layout = wgpu_bind_group_layout_compute!(
            dev,
            [
                (0, true),
                (1, true),
                (2, true),
                (3, false),
                (4, true),
                (5, true)
            ]
        );
        let compute_pipeline_layout = wgpu_compute_pipeline_layout!(dev, &[&bind_group_layout]);
        let pipeline = wgpu_compute_pipeline!(dev, &compute_pipeline_layout, &cs_module, "main");

        let start = std::time::Instant::now();
        println!("n_blocks: {}\tm_blocks: {}", n_blocks, m_blocks);
        for pass in 0..(n_blocks + m_blocks - 1) as u32 {
            let max_diag = if pass < n_blocks {
                min!(pass + 1, m_blocks)
            } else {
                min!(pass - n_blocks + 1, n_blocks)
            };
            // let num_cells = (0..max_diag)
            //     .map(|i| {
            //         let x = pass - i - (max!(0, 1 + pass as i32 - n_blocks as i32) as u32);
            //         let y = i + (max!(0, pass as i32 - n_blocks as i32) as u32);
            //         cell_counts[y as usize][x as usize]
            //     })
            //     .sum::<u32>();
            let num_cells = max_diag * WG_COLS * WG_ROWS;
            if PRINT {
                println!(
                    "pass: {}\tnum_cells: {}\tmax_diag: {}",
                    pass, num_cells, max_diag
                );
            }

            let pass_bytes = wbyte_of!(&pass);
            let max_diag_bytes = wbyte_of!(&max_diag);

            let pass_buffer = wgpu_buf_init!(dev, pass_bytes, [STORAGE | COPY_SRC]);
            let max_diag_buffer = wgpu_buf_init!(dev, max_diag_bytes, [STORAGE | COPY_SRC]);

            let bind_group = wgpu_bind_group!(
                dev,
                &bind_group_layout,
                [
                    (0, info_buf.as_entire_binding()),
                    (1, seq1_buf.as_entire_binding()),
                    (2, seq2_buf.as_entire_binding()),
                    (3, scores_buf.as_entire_binding()),
                    (4, pass_buffer.as_entire_binding()),
                    (5, max_diag_buffer.as_entire_binding())
                ]
            );
            let mut encoder = state.device.create_command_encoder(&Default::default());
            ts_query.write(&mut encoder);
            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(num_cells, 1, 1);
            }
            ts_query.write(&mut encoder);
            state.queue.submit(Some(encoder.finish()));

            {
                // Time the compute shader
                let (query_count, query_slice) = ts_query.map_async();
                state.device.poll(wgpu::Maintain::Wait);
                let ts_data = WTsQueryState::read(query_count, query_slice);
                let ts_period = state.queue.get_timestamp_period();
                if PRINT {
                    println!(
                        "compute shader elapsed: {:?}ms",
                        (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
                    );
                }
                ts_query.reset();
            }

            if PRINT {
                let mut encoder = state.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(
                    &scores_buf,
                    0,
                    &out_buf,
                    0,
                    score_bytes.len() as u64,
                );
                state.queue.submit(Some(encoder.finish()));

                let out_slice = out_buf.slice(..);
                let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                out_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

                // More queries
                state.device.poll(wgpu::Maintain::Wait);
                if let Some(Ok(())) = receiver.receive().await {
                    let data_raw = &*out_slice.get_mapped_range();
                    let data: &[Score] = bytemuck::cast_slice(data_raw);
                    print_scores(data, n as usize + 1, m as usize + 1);
                }
                out_buf.unmap();
                println!();
            }
        }
        println!("GPU took: {:?}", start.elapsed());

        let mut encoder = state.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&scores_buf, 0, &out_buf, 0, score_bytes.len() as u64);
        state.queue.submit(Some(encoder.finish()));

        let out_slice = out_buf.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        out_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // More queries
        state.device.poll(wgpu::Maintain::Wait);
        println!("\nFinal pass:");
        if let Some(Ok(())) = receiver.receive().await {
            let data_raw = &*out_slice.get_mapped_range();
            let data: &[Score] = bytemuck::cast_slice(data_raw);
            print_scores(data, n as usize + 1, m as usize + 1);
            return data[data.len() - 1];
        }

        todo!()
    }
}

mod tests {
    use super::*;

    #[allow(dead_code)]
    struct NeedlemanWunschData {
        seq1: Sequence,
        seq2: Sequence,
        gap_penalty: i32,
        mismatch_penalty: i32,
        match_score: i32,
        result: i32,
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
