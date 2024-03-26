use rustwarp::device::core::*;
use rustwarp::{typenum, wvec2, wvec_call};

#[repr(C)]
#[derive(Copy, Debug, Default, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct Test {
    t1: wvec2!(f32, 16),
}

fn main() {
    let x = vec![Test::default(); 10];
    dbg!(x);
    // let bytes: &[u8] = bytemuck::cast_slice(&x);

    // cpu_transform::run_test_transform();
    // pollster::block_on(device::setup::State::new());
    // let x = size_of!(device::core::Pix);
    // dbg!(x);
}
