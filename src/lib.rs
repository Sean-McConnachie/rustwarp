use std::ops::{Add, Div, Mul, Sub};

pub trait Numeric:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Copy
{
}

impl<T: Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + Copy> Numeric
    for T
{
}

pub mod vec2 {
    use crate::Numeric;

    pub type Matrix<T> = [[T; 3]; 3];

    #[derive(Clone, Copy, Debug)]
    pub struct Vec2<T> {
        pub x: T,
        pub y: T,
    }

    impl<T> Vec2<T> {
        pub const fn new(x: T, y: T) -> Self {
            Self { x, y }
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Vec2H<T> {
        pub x: T,
        pub y: T,
        pub w: T,
    }

    impl<T: Numeric> Vec2H<T>
    where
        T: From<u8>,
    {
        pub const fn new(x: T, y: T, w: T) -> Self {
            Self { x, y, w }
        }

        pub fn normalize(&mut self) {
            self.x = self.x / self.w;
            self.y = self.y / self.w;
            self.w = T::from(1);
        }

        pub fn transform(self, m: &Matrix<T>) -> Self {
            let x = self.x * m[0][0] + self.y * m[0][1] + self.w * m[0][2];
            let y = self.x * m[1][0] + self.y * m[1][1] + self.w * m[1][2];
            let w = self.x * m[2][0] + self.y * m[2][1] + self.w * m[2][2];
            let mut s = Self::new(x, y, w);
            s.normalize();
            s
        }
    }

    impl<T: Numeric> Into<Vec2H<T>> for &Vec2<T>
    where
        T: From<u8>,
    {
        fn into(self) -> Vec2H<T> {
            Vec2H::new(self.x, self.y, T::from(1))
        }
    }
}

pub mod col {
    use std::ops::Add;

    #[derive(Clone, Copy, Debug)]
    pub struct Pixel {
        pub r: u8,
        pub g: u8,
        pub b: u8,
    }

    impl Pixel {
        pub fn new(r: u8, g: u8, b: u8) -> Self {
            Self { r, g, b }
        }

        pub fn mult(&self, w: f64) -> Self {
            let m = |c| (c as f64 * w) as u8;
            Self::new(m(self.r), m(self.g), m(self.b))
        }
    }

    impl Add for Pixel {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            Self::new(
                self.r.saturating_add(rhs.r),
                self.g.saturating_add(rhs.g),
                self.b.saturating_add(rhs.b),
            )
        }
    }
}
