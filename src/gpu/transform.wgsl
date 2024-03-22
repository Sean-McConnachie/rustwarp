
struct ImageTransform {
    odim: vec2<u32>,
    tmatrix: mat3x3<f32>,
}

struct Image {
    colors: array<vec3<f32>>
}

@group(0)
@binding(0)
var<storage, read> transform: ImageTransform;

@group(0)
@binding(1)
var<storage, read_write> buf: array<vec3<f32>>;

// @group(0)
// @binding(0)
// var<storage, read> im_transform: ImageTransform;

// @group(0)
// @binding(1)
// var<storage, read> im_input: Image;

// @group(0)
// @binding(2)
// var<storage, write> im_output: Image;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var ind: u32;
    ind = global_id.x * transform.odim.x + global_id.y;
    var temp: vec3<f32>;
    temp = transform.tmatrix * buf[ind];
    buf[ind] = temp;
}