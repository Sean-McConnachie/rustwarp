use rustwarp::{
    modules::needleman_wunsch::{needleman_wunsch_gpu, Sequence},
    setup::WState,
};

#[tokio::main]
async fn main() {
    let seq1: Sequence = "ACTGATTCA".into();
    let seq2: Sequence = "ACGCATCA".into();
    let match_score = 2;
    let mismatch_penalty = -2;
    let gap_penalty = -2;
    let result = 8;

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
