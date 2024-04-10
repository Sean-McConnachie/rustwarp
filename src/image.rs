use bytemuck::{Pod, Zeroable};
use core::fmt;

use crate::tester::impl_prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod, PartialEq)]
pub struct Pix {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    // TODO
    pub _a: u8,
}

impl WTestable for Pix {
    fn wgsl_type() -> WType {
        WType::Primitive("u32")
    }
}

impl WDistribution<Pix> for WStandard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Pix {
        Pix {
            r: rng.gen(),
            g: rng.gen(),
            b: rng.gen(),
            _a: rng.gen(),
        }
    }
}

wtest!(Pix, 256);

impl Pix {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, _a: a }
    }

    pub fn mult(&self, v: f32) -> Self {
        Self {
            r: (self.r as f32 * v) as u8,
            g: (self.g as f32 * v) as u8,
            b: (self.b as f32 * v) as u8,
            _a: (self._a as f32 * v) as u8,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            r: self.r.saturating_add(other.r),
            g: self.g.saturating_add(other.g),
            b: self.b.saturating_add(other.b),
            _a: self._a.saturating_add(other._a),
        }
    }
}

impl fmt::Display for Pix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.r, self.g, self.b)
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

#[derive(Clone, Debug)]
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
            *pixel = image::Rgb([p.r.try_into()?, p.g.try_into()?, p.b.try_into()?]);
        }
        Ok(im)
    }
}
