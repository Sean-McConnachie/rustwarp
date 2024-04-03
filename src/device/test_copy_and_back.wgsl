
struct ImageTransform {
    odim: vec2<u32>,
    tmatrix: mat3x3<f32>,
}

// TODO: Change to uniform?
@group(0)
@binding(0)
var<storage, read> transform: ImageTransform;

@group(0)
@binding(1)
var<storage, read> input: array<vec3<u32>>;

@group(0)
@binding(2)
var<storage, read_write> output: array<vec3<u32>>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var dst_ind: u32;
    dst_ind = global_id.y * transform.odim.x + global_id.x;
    // dst_ind = global_id.x;

    output[dst_ind] = input[dst_ind];
}