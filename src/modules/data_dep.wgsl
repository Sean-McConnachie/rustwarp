@group(0)
@binding(0)
var<storage, read> rpass: u32;

@group(0)
@binding(1)
var<storage, read_write> input: array<i32>;

let UNDEFINED = -9999i;
// let workgroup_len = 3i;
var<workgroup> iter: u32 = 0u;
var<workgroup> workgroup_data: array<i32, 4>;

@compute
@workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var found = false;
    if lid.x == 0u {
        iter = 1u;
        workgroup_data[0u] = 1i;
        found = true;
    } else {
        workgroup_data[lid.x] = UNDEFINED;
    }

    workgroupBarrier();
    for (var i: u32 = 1u; i < 4u; i = i + 1u) {
        if workgroup_data[i - 1u] != UNDEFINED && !found && i >= lid.x {
            workgroup_data[lid.x] = workgroup_data[i - 1u] + 1i;
            found = true;
        }
        workgroupBarrier();
    }
    input[gid.x] = workgroup_data[lid.x];
}

