/*
use rustwarp::{modules::data_dep::data_dep_gpu, setup::WState};

#[tokio::main]
async fn main() {
    let mut state = WState::new().await;
    data_dep_gpu(&mut state).await;
}
*/
// /*
use rustwarp::{
    modules::needleman_wunsch::{needleman_wunsch_cpu, needleman_wunsch_gpu, Sequence},
    setup::WState,
};

#[tokio::main]
async fn main() {
    let seq1: Sequence = "AC".into();
    let seq2: Sequence = "AC".into();
    let match_score = 2;
    let mismatch_penalty = -2;
    let gap_penalty = -2;
    let result = needleman_wunsch_cpu(&seq1, &seq2, gap_penalty, mismatch_penalty, match_score);

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
    println!("GPU score: {}", score);
    assert_eq!(score, result);
}
// */
