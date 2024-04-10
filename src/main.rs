use futures::future::join_all;
use rustwarp::{
    core::WMat,
    image::{Image, Pix, Size},
    modules::multiple_warp::{
        warp_perspective_gpu, ImageTransform, Interpolation, WarpPerspectiveGpu,
    },
    setup::WState,
};

type WM = WMat<f32, 3, 3, 4>;

type M = [[f32; 3]; 3];
const IDENTITY: M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const IM_COUNT: usize = 4;
const PROC_COUNT: usize = 1;

#[tokio::main]
async fn main() {
    let size = Size::new(1920, 1080);
    let transforms = (0..IM_COUNT)
        .map(|i| {
            let mut m = IDENTITY;
            m[0][2] = 120.0 * i as f32;
            m[1][2] = 60.0 * i as f32;
            ImageTransform::new(size, WM::from_row_major(m))
        })
        .collect::<Vec<_>>();

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
    let inputs = (0..IM_COUNT).map(|_| input.clone()).collect::<Vec<_>>();
    let gpu_outputs = (0..IM_COUNT).map(|_| Image::new(size)).collect::<Vec<_>>();

    let mut states = join_all(
        (0..PROC_COUNT)
            .map(|_| async {
                let state = WState::new().await;
                WarpPerspectiveGpu::new(
                    state,
                    transforms.clone(),
                    Interpolation::None,
                    inputs.clone(),
                    gpu_outputs.clone(),
                )
            })
            .collect::<Vec<_>>(),
    )
    .await;

    // create futures for each state
    {
        let start = std::time::Instant::now();
        let futures = states
            .iter_mut()
            .map(|state| state.render_pass())
            .collect::<Vec<_>>();

        // run all futures in parallel
        join_all(futures).await;
        let elapsed = start.elapsed();
        println!("Total Elapsed: {:?}", elapsed);
    }

    let _: Vec<_> = (0..PROC_COUNT)
        .map(|i| {
            (0..IM_COUNT).for_each(|j| {
                let _ = states[i].dst[j]
                    .rgb_image()
                    .unwrap()
                    .save(format!("outputs/{}gpuoutput{}.png", i, j));
            });
        })
        .collect();
}
