
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
    var pos: vec3<f32>;
    pos.x = f32(global_id.x);
    pos.y = f32(global_id.y);
    pos.z = 1.0;

    pos = transform.tmatrix * pos;

    pos.x /= pos.z;
    pos.y /= pos.z;

    pos.x = max(0.0, min(pos.x, f32(transform.odim.x)));
    pos.y = max(0.0, min(pos.y, f32(transform.odim.y)));

    var src_ind: u32;
    // src_ind = u32(pos.y * f32(transform.odim.x) + pos.x);
    src_ind = u32(pos.y) * transform.odim.x + u32(pos.x);

    var dst_ind: u32;
    dst_ind = global_id.y * transform.odim.x + global_id.x;

    output[dst_ind] = input[src_ind];
}