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
var<storage, read_write> b1: array<Score>;

@group(0)
@binding(4)
var<storage, read_write> b2: array<Score>;

@group(0)
@binding(5)
var<storage, read_write> b3: array<Score>;

@group(0)
@binding(6)
var<storage, read> rpass: u32;

@group(0)
@binding(7)
var<storage, read> max_i: u32;

fn set_score(i: u32, v: Score) {
    if rpass % 3u == 0u {
        b1[i] = v;
    } else if rpass % 3u == 1u {
        b2[i] = v;
    } else {
        b3[i] = v;
    }
}


// Score 1 pass behind
fn get_score_m1(i: i32, offset: i32) -> Score {
    if rpass % 3u == 0u {
        return b1[u32(i + offset)];
    } else if rpass % 3u == 1u {
        return b3[u32(i + offset)];
    } else {
        return b2[u32(i + offset)];
    }
}

// Score 2 passes behind
fn get_score_m2(i: i32, offset: i32) -> Score {
    if rpass % 3u == 0u {
        return b3[u32(i + offset)];
    } else if rpass % 3u == 1u {
        return b2[u32(i + offset)];
    } else {
        return b1[u32(i + offset)];
    }
}

// // Score 1 pass behind
// fn get_score_m1(i: i32, offset: i32) -> Score {
//     if rpass % 3u == 0u {
//         return b2[u32(i + offset)];
//     } else if rpass % 3u == 1u {
//         return b3[u32(i + offset)];
//     } else {
//         return b1[u32(i + offset)];
//     }
// }

// // Score 2 passes behind
// fn get_score_m2(i: i32, offset: i32) -> Score {
//     if rpass % 3u == 0u {
//         return b3[u32(i + offset)];
//     } else if rpass % 3u == 1u {
//         return b1[u32(i + offset)];
//     } else {
//         return b2[u32(i + offset)];
//     }
// }

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i: u32;
    if rpass > info.seq2_len - 1u {
        i = global_id.x + rpass - info.seq2_len + 1u;
    } else {
        i = global_id.x;
    }

    if i == 0u {
        set_score(0u, i32(rpass) * info.gap_penalty);
        return;
    }
    if max_i == i && rpass <= info.seq1_len {
        // cannot take from left
        set_score(i, i32(rpass) * info.gap_penalty);
        return;
    }

    let sc_left = get_score_m1(i32(i), 1i) + info.gap_penalty;
    let sc_above = get_score_m1(i32(i), -1i) + info.gap_penalty;

    var sc_align: Score;
    let s1k = global_id.x;
    let s2k = rpass - global_id.x - 2u;
    sc_align = get_score_m2(i32(i), 0i);
    if seq1[s1k] == seq2[s2k] {
        sc_align += info.match_score;
    } else {
        sc_align += info.mismatch_penalty;
    }

    // set_score(0u, -2147483648i);
    // set_score(1u, -2147483648i);
    // set_score(0u, i32(s1k));
    // set_score(0u, i32(i));
    set_score(i, max(sc_left, max(sc_above, sc_align)));
}
