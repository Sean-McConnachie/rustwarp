use bytemuck::{Pod, Zeroable};
use rustwarp::*;

wvec_impls!(
    struct Temp {
        v0: wvec2!(bool, 14), // 16 bytes type(2 * 1) + pad(14 * 1)
        v1: wvec3!(f32, 4),   // 16 bytes type(3 * 4) + pad(4 * 1)
        v2: wvec4!(f32, 0),   // 16 bytes type(4 * 4) + pad(0 * 1)
    }
);

fn main() {
    let mut temp = vec![Temp::default(); 2];
    temp[0].v0.x = true;
    temp[1].v1.z = 1.234;
    let bytes: &[u8] = bytemuck::cast_slice(&temp);
    println!("{:?}", bytes);
    let y: &[Temp] = bytemuck::try_cast_slice(bytes).unwrap();
    println!("{:?}", y);

    assert!(temp[0].v0.x == y[0].v0.x);
    assert!(temp[1].v1.z == y[1].v1.z);
}
