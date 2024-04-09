use bytemuck::{Pod, Zeroable};
use core::fmt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct Pix {
    pub colour: wvec3!(u32, 4),
}

impl Pix {
    pub fn new(r: u32, g: u32, b: u32) -> Self {
        let mut p = Self::default();
        p.colour.x = r;
        p.colour.y = g;
        p.colour.z = b;
        p
    }

    pub fn mult(&self, v: f32) -> Self {
        let mut s = Self::clone(&self);
        s.colour.x = (s.colour.x as f32 * v) as u32;
        s.colour.y = (s.colour.y as f32 * v) as u32;
        s.colour.z = (s.colour.z as f32 * v) as u32;
        s
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut s = Self::clone(&self);
        s.colour.x += other.colour.x;
        s.colour.y += other.colour.y;
        s.colour.z += other.colour.z;
        s
    }
}

impl fmt::Display for Pix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {})",
            self.colour.x, self.colour.y, self.colour.z
        )
    }
}

pub type Pos = Point;
pub type Size = Point;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

impl Point {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

pub struct Image {
    pub data: Vec<Pix>,
    pub size: Size,
}

impl Image {
    pub fn new(size: Size) -> Self {
        Self {
            data: vec![Pix::default(); size.x * size.y],
            size,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &Pix {
        &self.data[y * self.size.x as usize + x]
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut Pix {
        &mut self.data[y * self.size.x as usize + x]
    }

    pub fn rgb_image(&self) -> Result<image::RgbImage, Box<dyn std::error::Error>> {
        TryInto::<image::RgbImage>::try_into(self)
    }
}

impl TryInto<image::RgbImage> for &Image {
    type Error = Box<dyn std::error::Error>;
    fn try_into(self) -> Result<image::RgbImage, Self::Error> {
        let mut im = image::RgbImage::new(self.size.x as u32, self.size.y as u32);
        for (x, y, pixel) in im.enumerate_pixels_mut() {
            let p = self.get(x as usize, y as usize);
            *pixel = image::Rgb([
                p.colour.x.try_into()?,
                p.colour.y.try_into()?,
                p.colour.z.try_into()?,
            ]);
        }
        Ok(im)
    }
}
