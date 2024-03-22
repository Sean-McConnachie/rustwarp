use rustwarp::*;

fn main() {
    // cpu_transform::run_test_transform();
    pollster::block_on(gpu::setup::State::new());
}
