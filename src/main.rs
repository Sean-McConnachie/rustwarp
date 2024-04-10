use pollster::FutureExt;
use rand::distributions::Distribution;
use rustwarp::{
    core::WMat,
    image::{Image, Pix, Size},
    modules::warp_perspective::{
        warp_perspective_cpu, warp_perspective_gpu, ImageTransform, Interpolation,
    },
    setup::WState,
    tester::{self, WTestable, WType},
    wpad, wtest, wvec3,
};

type M = [[f32; 3]; 3];
type WM = WMat<f32, 3, 3, 4>;
#[allow(unused)]
const IDENTITY: M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
#[allow(unused)]
const SCALE_XY: M = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
#[allow(unused)]
const TRANSLATE_XY: M = [[1.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

const ROTATE: M = [
    [0.9594929736144975, 0.28173255684142967, 0.0],
    [-0.2817325568414297, 0.9594929736144975, 0.0],
    [0.0, 0.0, 1.0],
];

const MAT: M = ROTATE;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Zeroable, bytemuck::Pod, PartialEq)]
struct MyStruct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl WTestable for MyStruct {
    fn wgsl_type() -> WType {
        WType::Primitive("u32")
    }
}

impl Distribution<MyStruct> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> MyStruct {
        let mut r = MyStruct::default();
        r.r = rng.gen();
        r.g = rng.gen();
        r.b = rng.gen();
        r.a = rng.gen();
        r
    }
}

wtest!(MyStruct, 256);

fn main() {
    let mut state = WState::new().block_on();

    let mat = WM::from_row_major(MAT);
    let mat = mat.inverse();
    println!("{}", &mat);
    let size = Size::new(1920, 1080);
    let input = {
        let mut im = Image::new(size);
        const BOX_SIZE: usize = 200 / 2;
        let hw = size.x / 2;
        let hh = size.y / 2;
        let total = BOX_SIZE * 8;
        let mut c = 0;
        let mut set_pix = |pix: &mut Pix| {
            pix.r = 255;
            pix.g = 255;
            let z = ((c as f32 / total as f32) * 255.0) as u32;
            let z = z.min(255);
            pix.b = z as u8;
            c += 1;
        };
        for y in hh - BOX_SIZE..hh + BOX_SIZE {
            set_pix(im.get_mut(hw - BOX_SIZE, y));
            set_pix(im.get_mut(hw + BOX_SIZE, y));
        }
        for x in hw - BOX_SIZE..hw + BOX_SIZE {
            set_pix(im.get_mut(x, hh - BOX_SIZE));
            set_pix(im.get_mut(x, hh + BOX_SIZE));
        }
        im
    };

    let _ = input
        .rgb_image()
        .unwrap()
        .save("outputs/input.png")
        .unwrap();

    let transform = ImageTransform::new(size, mat);

    let mut cpu_output = Image::new(size);
    warp_perspective_cpu(&transform, Interpolation::Bilinear, &input, &mut cpu_output);
    let _ = cpu_output
        .rgb_image()
        .unwrap()
        .save("outputs/cpuoutput.png")
        .unwrap();

    let mut gpu_output = Image::new(size);
    warp_perspective_gpu(
        &mut state,
        &transform,
        Interpolation::Bilinear,
        &input,
        &mut gpu_output,
    )
    .block_on();
    let _ = gpu_output
        .rgb_image()
        .unwrap()
        .save("outputs/gpuoutput.png")
        .unwrap();
}
