
@group(0)
@binding(0)
var<storage, read> rpass: u32;

@group(0)
@binding(1)
var<storage, read_write> input: array<i32>;

const workgroup_len: u32 = 20;
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
@workgroup_size(workgroup_len, 1, 1) // must match workgroup_len
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if lid.x == 0u {
        workgroup_data[lid.x] = 0i;
    } else if lid.x == 1u {
        workgroup_data[lid.x] = 1i;
    }

    spin(lid.x);
    if lid.x >= 2u {
        workgroup_data[lid.x] = workgroup_data[lid.x - 1u] + workgroup_data[lid.x - 2u];
    }

    input[gid.x] = workgroup_data[lid.x];
    

    // if lid.x % workgroup_len != 0u {
    //     spin(lid.x);
    //     workgroup_data[lid.x] = workgroup_data[lid.x - 1u] + 1i;
    // } else {
    //     var x: u32 = lid.x;
    //     for (var i: u32 = 1; i < lid.x + 2u; i += 1u) {
    //         x += i;
    //     }
    //     workgroup_data[lid.x] = i32(x);
    // }


    // input[gid.x] = workgroup_data[lid.x];
}

