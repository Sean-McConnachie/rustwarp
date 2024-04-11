// Seq1 = column
// Seq2 = row
struct NWInfo {
    seq1_len: u32,
    seq2_len: u32,
    gap_penalty: i32,
    mismatch_penalty: i32,
    match_score: i32,
}

type Nucleotite = u32;
type Score = i32;

let UNDEFINED: Score = -2147483648i;

@group(0)
@binding(0)
var<storage, read> info: NWInfo;

@group(0)
@binding(1)
var<storage, read> seq1: array<Nucleotite>;

@group(0)
@binding(2)
var<storage, read> seq2: array<Nucleotite>;

@group(0)
@binding(3)
var<storage, read_write> alignment: array<Score>;

fn alignment_score(x: u32, y: u32) -> Score {
    return alignment[y * info.seq2_len + x];
}

fn set_alignment_score(x: u32, y: u32, score: Score) {
    alignment[y * info.seq2_len + x] = score;
}

fn spin(x: u32, y: u32) -> Score {
    let ind = y * info.seq2_len + x;
    var score: Score;
    loop {
        score = alignment[ind];
        if score != UNDEFINED {
            return score;
        }
    }
    return 0i;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if i == 0u || j == 0u {
        return;
    }

    let i1j1 = spin(i - 1u, j - 1u);
    let i1j0 = spin(i - 1u, j);
    let i0j1 = spin(i, j - 1u);

    var score: Score;
    if seq1[i - 1u] == seq2[j - 1u] {
        score = info.match_score;
    } else {
        score = info.mismatch_penalty;
    }

    let best = max(i1j1 + score, max(i1j0 + info.gap_penalty, i0j1 + info.gap_penalty));
    set_alignment_score(i, j, best);
}
