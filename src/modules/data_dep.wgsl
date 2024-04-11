@group(0)
@binding(0)
var<storage, read_write> input: array<i32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    input[gid.x] = input[gid.x] + 1i;
}