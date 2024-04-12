// /*
use rustwarp::{modules::data_dep::data_dep_gpu, setup::WState};

#[tokio::main]
async fn main() {
    let mut state = WState::new().await;
    data_dep_gpu(&mut state).await;
}
// */
/*
use rustwarp::{
    modules::needleman_wunsch::{needleman_wunsch_cpu, needleman_wunsch_gpu, Nucleotide, Sequence},
    setup::WState,
};

#[tokio::main]
async fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let seq1: Sequence = Sequence(
        (0..2048)
            .map(|_| {
                let nuc: Nucleotide = rng.gen();
                nuc
            })
            .collect::<Vec<_>>(),
    );
    let seq2: Sequence = Sequence(
        (0..1024)
            .map(|_| {
                let nuc: Nucleotide = rng.gen();
                nuc
            })
            .collect::<Vec<_>>(),
    );
    // let seq1: Sequence = "ACGT".into();
    // let seq2: Sequence = "ACG".into();
    let match_score = 2;
    let mismatch_penalty = -2;
    let gap_penalty = -2;

    let start = std::time::Instant::now();
    let result = needleman_wunsch_cpu(&seq1, &seq2, gap_penalty, mismatch_penalty, match_score);
    println!("CPU took: {:?}", start.elapsed());

    let mut state = WState::new().await;
    let score = needleman_wunsch_gpu(
        &mut state,
        &seq1,
        &seq2,
        gap_penalty,
        mismatch_penalty,
        match_score,
    )
    .await;

    assert_eq!(score, result);
    println!("GPU and CPU results are equal!")
}
*/
