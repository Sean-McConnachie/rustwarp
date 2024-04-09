
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

fn ind(x: u32, y: u32, width: u32) -> u32 {
    return y * width + x;
}

@compute
@workgroup_size(1)
fn interpolation_none(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pos = vec3<f32>(f32(global_id.x), f32(global_id.y), 1.0);
    pos = transform.tmatrix * pos;
    pos /= pos.z;

    if pos.x < 0.0 || pos.x >= f32(transform.odim.x) || pos.y < 0.0 || pos.y >= f32(transform.odim.y) {
        return;
    }

    let src_ind = ind(u32(pos.x), u32(pos.y), transform.odim.x);
    let dst_ind = ind(global_id.x, global_id.y, transform.odim.x);

    output[dst_ind] = input[src_ind];
}

@compute
@workgroup_size(1)
fn interpolation_bilinear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Floating point position
    var fpos = vec3<f32>(f32(global_id.x), f32(global_id.y), 1.0);
    fpos = transform.tmatrix * fpos;
    fpos /= fpos.z;

    if fpos.x < 0.0 || fpos.x >= f32(transform.odim.x) || fpos.y < 0.0 || fpos.y >= f32(transform.odim.y) {
        return;
    }

    // Floored floating point position
    let rpos = vec2<f32>(floor(fpos.x), floor(fpos.y));
    //  Floored integer position
    let ipos = vec2<u32>(u32(rpos.x), u32(rpos.y));
    // Fractional part
    let fpart = vec2<f32>(fpos.x - rpos.x, fpos.y - rpos.y);

    let p0 = (1.0 - fpart.x) * (1.0 - fpart.y) * vec3<f32>(input[ind(ipos.x, ipos.y, transform.odim.x)]);
    let p1 = fpart.x * (1.0 - fpart.y) * vec3<f32>(input[ind(ipos.x + 1u, ipos.y, transform.odim.x)]);
    let p2 = (1.0 - fpart.x) * fpart.y * vec3<f32>(input[ind(ipos.x, ipos.y + 1u, transform.odim.x)]);
    let p3 = fpart.x * fpart.y * vec3<f32>(input[ind(ipos.x + 1u, ipos.y + 1u, transform.odim.x)]);

    output[ind(global_id.x, global_id.y, transform.odim.x)] = vec3<u32>(p0 + p1 + p2 + p3);
}