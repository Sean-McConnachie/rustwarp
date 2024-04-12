// Seq1 = column
// Seq2 = row
struct NWInfo {
    seq1_len: u32,
    seq2_len: u32,
    gap_penalty: i32,
    mismatch_penalty: i32,
    match_score: i32,
}

alias Nucleotite = u32;
alias Score = i32;

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
var<storage, read_write> scores: array<Score>;

@group(0)
@binding(4)
var<storage, read> rpass: u32;

const WG_COLS: u32 = {{WG_COLS}};
const WG_ROWS: u32 = {{WG_ROWS}};
const WG_SHARED_SIZE: u32 = WG_COLS * WG_ROWS;

var<workgroup> block: array<Score, WG_SHARED_SIZE>;

fn get_score(x: u32, y: u32) -> Score {
    return scores[y * (info.seq2_len + 1u) + x];
}

fn set_score(x: u32, y: u32, score: Score) {
    scores[y * (info.seq2_len + 1u) + x] = score;
}

@compute
@workgroup_size(WG_COLS, WG_ROWS, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    let x = rpass - i - u32(max(0i, i32(rpass) - i32(info.seq2_len)));
    let y = u32(max(0i, i32(rpass) - i32(info.seq2_len))) + i;

    if x == 0u {
        set_score(x, y, i32(y) * info.gap_penalty);
        return;
    }
    if y == 0u {
        set_score(x, y, i32(x) * info.gap_penalty);
        return;
    }

    var match_score: i32;
    if seq1[y - 1u] == seq2[x - 1u] {
        match_score = info.match_score;
    } else {
        match_score = info.mismatch_penalty;
    }

    let up = get_score(x, y - 1u) + info.gap_penalty;
    let left = get_score(x - 1u, y) + info.gap_penalty;
    let diag = get_score(x - 1u, y - 1u) + match_score;

    let score = max(max(up, left), diag);
    set_score(x, y, score);
}