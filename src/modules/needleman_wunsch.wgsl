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

@group(0)
@binding(5)
var<storage, read> max_diag: u32;

const WG_COLS: u32 = {{WG_COLS}};
const WG_ROWS: u32 = {{WG_ROWS}};
const WG_SIZE: u32 = WG_COLS * WG_ROWS;

var<workgroup> block: array<Score, WG_SIZE>;

fn get_global_score(x: u32, y: u32) -> Score {
    return scores[y * (info.seq2_len + 1u) + x];
}

fn set_global_score(x: u32, y: u32, score: Score) {
    scores[y * (info.seq2_len + 1u) + x] = score;
}

fn get_local_score(x: u32, y: u32) -> Score {
    return block[y * WG_COLS + x];
}

fn set_local_score(x: u32, y: u32, score: Score) {
    block[y * WG_COLS + x] = score;
}

fn spin(k: u32) {
    for (var i: u32 = 0u; i < 999999u; i = i + 1u) {
        if i == k {
            return;
        }
        if i == 1u {
            workgroupBarrier();
            workgroupBarrier();
        }
        if i == 2u {
            workgroupBarrier();
        }
    }
}

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let gbl_i = wid.x;

    let blk_x = rpass - gbl_i - u32(max(0i, 1i + i32(rpass) - i32(info.seq2_len)));
    let blk_y = gbl_i + u32(max(0i, i32(rpass) - i32(info.seq2_len)));

    let lcl_x = lid.x % WG_COLS;
    let lcl_y = lid.x / WG_COLS;
    let lcl_i = u32(-(-i32(lcl_y) - i32(lcl_x)));

    let gbl_x = blk_x * WG_COLS + lcl_x;
    let gbl_y = blk_y * WG_ROWS + lcl_y;

    let max_x = info.seq2_len; // + 1 - 1
    let max_y = info.seq1_len; // + 1 - 1

    if gbl_x > max_x || gbl_y > max_y {
        return;
    }

    var up: Score;
    var left: Score;
    var up_left: Score;

    // if gbl_x == 0u {
    //     let penalty = i32(gbl_y) * info.gap_penalty;
    //     set_local_score(lcl_x, lcl_y, penalty);
    //     if lcl_y == WG_ROWS - 1u {
    //         set_global_score(gbl_x, gbl_y, penalty);
    //     }
    //     return;
    // } else if gbl_y == 0u {
    //     let penalty = i32(gbl_x) * info.gap_penalty;
    //     set_local_score(lcl_x, lcl_y, penalty);
    //     if lcl_x == WG_COLS - 1u {
    //         set_global_score(gbl_x, gbl_y, penalty);
    //     }
    //     return;
    // }

    spin(lcl_i);

    if gbl_x != 0u && gbl_y != 0u {
        if lcl_x == 0u && lcl_y == 0u {
            up = get_global_score(gbl_x, gbl_y - 1u);
            left = get_global_score(gbl_x - 1u, gbl_y);
            up_left = get_global_score(gbl_x - 1u, gbl_y - 1u);
        } else if lcl_x == 0u {
            up = get_local_score(lcl_x, lcl_y - 1u);
            left = get_global_score(gbl_x - 1u, gbl_y);
            up_left = get_global_score(gbl_x - 1u, gbl_y - 1u);
        } else if lcl_y == 0u {
            up = get_global_score(gbl_x, gbl_y - 1u);
            left = get_local_score(lcl_x - 1u, lcl_y);
            up_left = get_global_score(gbl_x - 1u, gbl_y - 1u);
        } else {
            up = get_local_score(lcl_x, lcl_y - 1u);
            left = get_local_score(lcl_x - 1u, lcl_y);
            up_left = get_local_score(lcl_x - 1u, lcl_y - 1u);
        }

        var match_score: i32;
        if seq1[gbl_y - 1u] == seq2[gbl_x - 1u] {
            match_score = info.match_score;
        } else {
            match_score = info.mismatch_penalty;
        }
        let score = max(up_left + match_score, max(up + info.gap_penalty, left + info.gap_penalty));

        set_local_score(lcl_x, lcl_y, score);
        set_global_score(gbl_x, gbl_y, score);
        set_global_score(0u, 0u, i32(lcl_i)); // should be 2 but it's 1

    // if lcl_x == WG_COLS - 1u || lcl_y == WG_ROWS - 1u || gbl_x == max_x || gbl_y == max_y {
    //     set_global_score(gbl_x, gbl_y, score);
    // }
    }
}