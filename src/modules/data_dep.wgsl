@group(0)
@binding(0)
var<storage, read> rpass: u32;

@group(0)
@binding(1)
var<storage, read_write> input: array<i32>;

let workgroup_len = 64i;
var<workgroup> workgroup_data: array<i32, workgroup_len>;

fn spin(k: u32) {
    for (var i: u32 = 0u; i < u32(workgroup_len); i = i + 1u) {
        if i == k {
            return;
        }
        workgroupBarrier();
    }
}

@compute
@workgroup_size(64, 1, 1) // must match workgroup_len
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    spin(lid.x);
    if lid.x == 0u {
        workgroup_data[0u] = 1i;
    }
    workgroup_data[lid.x] = workgroup_data[lid.x - 1u] + 1i;

    input[gid.x] = workgroup_data[lid.x];
}

