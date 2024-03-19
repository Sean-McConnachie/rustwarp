use rustwarp::col::*;
use rustwarp::vec2::*;

type Flt = f64;
type VecIm = Vec<Vec<Pixel>>;

const SIZE: Vec2<u32> = Vec2::new(800, 600);

const TRANSFORM: Matrix<Flt> = [
    [0.7071067811865476, -0.7071067811865475, 328.7893218813452],
    [0.7071067811865475, 0.7071067811865476, -194.76764004939668],
    [0.0, 0.0, 1.0],
];

fn print_matrix(m: &Matrix<Flt>) {
    for row in m.iter() {
        for cell in row.iter() {
            print!("{:.2}\t", cell);
        }
        println!();
    }
}

fn invert(m: &Matrix<Flt>) -> Matrix<Flt> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let t_a = e * i - f * h;
    let t_b = -(d * i - f * g);
    let t_c = d * h - e * g;
    let t_d = -(b * i - c * h);
    let t_e = a * i - c * g;
    let t_f = -(a * h - b * g);
    let t_g = b * f - c * e;
    let t_h = -(a * f - c * d);
    let t_i = a * e - b * d;

    let det = a * t_a + b * t_b + c * t_c;
    if det == 0 as Flt {
        panic!("Matrix is not invertible");
    }
    let det_inv = 1 as Flt / det;

    let m_inv = [
        [t_a * det_inv, t_d * det_inv, t_g * det_inv],
        [t_b * det_inv, t_e * det_inv, t_h * det_inv],
        [t_c * det_inv, t_f * det_inv, t_i * det_inv],
    ];
    m_inv
}

fn make_vec_image(size: Vec2<u32>) -> VecIm {
    let mut vec2d = Vec::new();
    for _ in 0..size.y {
        let row = vec![Pixel { r: 0, g: 0, b: 0 }; size.x as usize];
        vec2d.push(row);
    }
    vec2d
}

fn vec_image_to_rgbimage(vec2d: VecIm) -> image::RgbImage {
    let mut im = image::RgbImage::new(vec2d[0].len() as u32, vec2d.len() as u32);
    for (x, y, pixel) in im.enumerate_pixels_mut() {
        let p = vec2d[y as usize][x as usize];
        *pixel = image::Rgb([p.r, p.g, p.b]);
    }
    im
}

#[allow(dead_code)]
enum Interpolate {
    Bilinear,
    None,
}

/// This function transform src onto dst by using the inverse of the transformation matrix, then iterating over the destination points.
fn transform_via_dst(
    src: &VecIm,
    dst: &mut VecIm,
    m: &Matrix<Flt>,
    interp: Interpolate,
    m_is_inv: bool,
) {
    let m = match m_is_inv {
        true => m.clone(),
        false => invert(m),
    };

    let sdims = Vec2::new(src[0].len(), src.len());
    let ddims = Vec2::new(dst[0].len(), dst.len());

    for y in 0..ddims.y {
        for x in 0..ddims.x {
            let pi = Vec2::new(x, y);
            let pt = Vec2H::new(pi.x as Flt, pi.y as Flt, 1 as Flt).transform(&m);
            let pn = Vec2::new(pt.x.floor() as usize, pt.y.floor() as usize);

            match interp {
                Interpolate::None => {
                    if pn.x <= 0 || pn.x >= sdims.x || pn.y <= 0 || pn.y >= sdims.y {
                        dst[pi.y][pi.x] = Pixel::new(0, 0, 0);
                        continue;
                    }
                    dst[pi.y][pi.x] = src[pn.y][pn.x];
                }
                Interpolate::Bilinear => {
                    if pn.x <= 0 || pn.x >= sdims.x - 1 || pn.y <= 0 || pn.y >= sdims.y - 1 {
                        dst[pi.y][pi.x] = Pixel::new(0, 0, 0);
                        continue;
                    }
                    let pd = Vec2::new(pt.x - pn.x as Flt, pt.y - pn.y as Flt);
                    dst[pi.y][pi.x] = src[pn.y][pn.x].mult((1 as Flt - pd.x) * (1 as Flt - pd.y))
                        + src[pn.y][pn.x + 1].mult(pd.x * (1 as Flt - pd.y))
                        + src[pn.y + 1][pn.x].mult((1 as Flt - pd.x) * pd.y)
                        + src[pn.y + 1][pn.x + 1].mult(pd.x * pd.y);
                }
            }
        }
    }
}

/// This function transforms src onto dst by iterating over the source points.
fn transform_via_src(src: &VecIm, dst: &mut VecIm, m: &Matrix<Flt>, interp: Interpolate) {
    let sdims = Vec2::new(src[0].len(), src.len());
    let ddims = Vec2::new(dst[0].len(), dst.len());

    assert!(sdims.x == ddims.x && sdims.y == ddims.y);

    for y in 0..sdims.y - 1 {
        for x in 0..sdims.x - 1 {
            let pi = Vec2::new(x, y);
            let pt = Vec2H::new(pi.x as Flt, pi.y as Flt, 1 as Flt).transform(m);
            let pn = Vec2::new(pt.x.floor() as usize, pt.y.floor() as usize);
            match interp {
                Interpolate::None => {
                    if pn.x <= 0 || pn.x >= sdims.x || pn.y <= 0 || pn.y >= sdims.y {
                        continue;
                    }
                    dst[pn.y][pn.x] = src[pi.y][pi.x];
                }
                Interpolate::Bilinear => {
                    if pn.x <= 0 || pn.x >= sdims.x - 1 || pn.y <= 0 || pn.y >= sdims.y - 1 {
                        continue;
                    }
                    let pd = Vec2::new(pt.x - pn.x as Flt, pt.y - pn.y as Flt);
                    dst[pn.y][pn.x] = src[pi.y][pi.x].mult((1 as Flt - pd.x) * (1 as Flt - pd.y))
                        + src[pi.y][pi.x + 1].mult(pd.x * (1 as Flt - pd.y))
                        + src[pi.y + 1][pi.x].mult((1 as Flt - pd.x) * pd.y)
                        + src[pi.y + 1][pi.x + 1].mult(pd.x * pd.y);
                }
            }
        }
    }
}

fn main() {
    let mut src = make_vec_image(SIZE);
    {
        let square_dist_from_orig = 100;
        for pt in [
            Vec2::new(
                SIZE.x / 2 - square_dist_from_orig,
                SIZE.y / 2 - square_dist_from_orig,
            ),
            Vec2::new(
                SIZE.x / 2 + square_dist_from_orig,
                SIZE.y / 2 - square_dist_from_orig,
            ),
            Vec2::new(
                SIZE.x / 2 - square_dist_from_orig,
                SIZE.y / 2 + square_dist_from_orig,
            ),
            Vec2::new(
                SIZE.x / 2 + square_dist_from_orig,
                SIZE.y / 2 + square_dist_from_orig,
            ),
        ]
        .iter()
        {
            let square_size = 10 / 2;
            for y in pt.y - square_size..pt.y + square_size {
                for x in pt.x - square_size..pt.x + square_size {
                    src[y as usize][x as usize] = Pixel::new(255, 255, 255);
                }
            }
        }
    }
    for x in 0..SIZE.x as usize / 2 {
        let pix = &mut src[SIZE.y as usize / 2][x];
        pix.r = 255;
        pix.g = 255;
        pix.b = 255;
    }

    let mut dst = make_vec_image(SIZE);
    // transform_via_src(&src, &mut dst, &TRANSFORM, Interpolate::Bilinear);
    transform_via_dst(&src, &mut dst, &TRANSFORM, Interpolate::Bilinear, false);

    vec_image_to_rgbimage(src)
        .save("outputs/input.png")
        .unwrap();
    vec_image_to_rgbimage(dst)
        .save("outputs/output.png")
        .unwrap();
}
