use core::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use bytemuck::{Pod, Zeroable};
use paste::paste;

// TODO: WScalars; AbstractInt, AbstractFloat, f16
pub trait WScalars: Copy + Zeroable {}
impl WScalars for bool {}
impl WScalars for u32 {}
impl WScalars for i32 {}
impl WScalars for f32 {}

pub trait WHostToDev {
    fn bytes(&self) -> &[u8];
}

pub trait WDevToHost {
    fn from_bytes_new(bytes: &[u8]) -> Self;
    fn from_bytes(&mut self, bytes: &[u8]);
}

#[macro_export]
macro_rules! wvec_def {
    (struct $struct_name:ident { $($field_name:ident : $field_type:ty),* $(,)? }) => {
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
        struct $struct_name {
            $( $field_name : $field_type ),*
        }
    };
}

#[macro_export]
macro_rules! wvec_impl {
    (block: $block:block) => {
        $block
    };
}

macro_rules! wvec_print_fields {
    ($f:expr, $s:expr, $first:ident, $($rest:tt),*) => {
        write!($f, "{}, ", $s.$first)?;
        wvec_print_fields!($f, $s, $($rest),*);
    };
    ($f:expr, $s:expr, $first:ident) => {
        write!($f, "{}", $s.$first)?;
    };
}

macro_rules! wvec_def_struct {
    ($struct:ident { $($fields:ident),* }, $pad:expr) => {
        paste! {
            #[repr(C)]
            #[derive(Copy, Clone, Debug, Default, Zeroable)]
            pub struct [< $struct P $pad >] <T>
            where
                T: WScalars,
            {
                $(pub $fields: T,)*
                _pad: [u8; $pad],
            }

            unsafe impl<T> Pod for [< $struct P $pad >]<T>
            where
                T: WScalars + 'static,
            { }

            impl<T> fmt::Display for [< $struct P $pad >]<T>
            where
                T: WScalars + fmt::Display
            {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}P{}(", stringify!($struct), stringify!($pad))?;
                    wvec_print_fields!(f, self, $($fields),*);
                    write!(f, ")")
                }
            }

            impl<T: WScalars> [< $struct P $pad >]<T> {
                pub fn new($($fields: T),*) -> Self {
                    Self {
                        $($fields,)*
                        _pad: [0; $pad],
                    }
                }

                pub fn set(&mut self, $($fields: T,)*) {
                    $(self.$fields = $fields;)*
                }
            }
        }
    };
}

macro_rules! wvec_def_struct_recursion {
    ($struct:ident { $($fields:ident),* }, [ $pad:expr, $($rem:tt),* ]) => {
        wvec_def_struct!($struct { $($fields),* }, $pad);
        wvec_def_struct_recursion!($struct { $($fields),* }, [ $($rem),* ]);
    };
    ($struct:ident { $($fields:ident),* }, [ $pad:expr ]) => {
        wvec_def_struct!($struct { $($fields),* }, $pad);
    };
}

macro_rules! wvec_def_macro_impl {
    ($struct:ident, $macro:ident, [ $($pad:expr),* ]) => {
        paste! {
            #[macro_export]
            macro_rules! $macro {
                $(
                    ($typ:ty, [< $pad >]) => {
                        $crate::core::[< $struct P $pad >]<$typ>
                    };
                )*
            }
        }
    };
}

macro_rules! wvec_def {
    ($struct:ident { $($fields:ident),* }, $macro:ident, [ $($pads:expr),* ]) => {
        wvec_def_struct_recursion!($struct { $($fields),* }, [ $($pads),* ]);
        wvec_def_macro_impl!($struct, $macro, [ $($pads),* ]);
    };
}

wvec_def!(
    WVec2 { x, y },
    wvec2,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
);

wvec_def!(
    WVec3 { x, y, z },
    wvec3,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
);

wvec_def!(
    WVec4 { x, y, z, w },
    wvec4,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
);

pub type WMat3x3Affine = WMat<f32, 3, 3, 4>;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable)]
pub struct WMat<T, const N: usize, const M: usize, const FORCED_M: usize>([[T; FORCED_M]; N])
where
    T: WScalars,
    [[T; FORCED_M]; N]: Default + Zeroable;

unsafe impl<T, const N: usize, const M: usize, const FORCED_M: usize> Pod
    for WMat<T, N, M, FORCED_M>
where
    T: WScalars + 'static,
    [[T; FORCED_M]; N]: Copy + Zeroable + Default,
{
}

impl<T: WScalars, const N: usize, const M: usize, const FORCED_M: usize> fmt::Display
    for WMat<T, N, M, FORCED_M>
where
    T: fmt::Display,
    [[T; FORCED_M]; N]: Default + Zeroable,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SPACING: usize = 15;
        write!(f, "[ ")?;
        for i in 0..N {
            for j in 0..M {
                if i != 0 && j == 0 {
                    write!(f, "  ")?;
                };
                write!(f, "{:width$}", self.0[i][j], width = SPACING)?;
            }
            if i != N - 1 {
                write!(f, "\n")?;
            }
        }
        write!(f, " ]")?;
        Ok(())
    }
}

// impl WMat
impl<T: WScalars, const N: usize, const M: usize, const FORCED_M: usize> WMat<T, N, M, FORCED_M>
where
    T: Default + Copy,
    [[T; FORCED_M]; N]: Default + Zeroable,
{
    pub fn from_row_major<const K: usize, const D: usize>(data: [[T; K]; D]) -> Self {
        assert_eq!(D, N);
        assert_eq!(K, M);
        let mut m = Self::default();
        for i in 0..N {
            for j in 0..M {
                m.0[i][j] = data[j][i];
            }
        }
        m
    }

    pub fn from_col_major<const K: usize, const D: usize>(data: [[T; K]; D]) -> Self {
        assert_eq!(K, N);
        assert_eq!(D, M);
        let mut m = Self::default();
        for i in 0..N {
            for j in 0..M {
                m.0[i][j] = data[i][j];
            }
        }
        m
    }

    pub fn set(&mut self, x: usize, y: usize, v: T) {
        assert!(x < M && y < N);
        self.0[x][y] = v;
    }

    pub fn get(&self, x: usize, y: usize) -> T {
        self.0[x][y]
    }

    pub fn matrix(&self) -> &[[T; FORCED_M]; N] {
        &self.0
    }

    pub fn matrix_mut(&mut self) -> &mut [[T; FORCED_M]; N] {
        &mut self.0
    }
}

impl<T: WScalars, const FORCED_M: usize> WMat<T, 3, 3, FORCED_M>
where
    T: Default
        + Mul<Output = T>
        + Sub<Output = T>
        + Neg<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + From<u8>
        + Copy
        + PartialEq,
    [[T; FORCED_M]; 3]: Default + Zeroable,
{
    pub fn try_inverse(&self) -> Result<Self, Box<dyn std::error::Error>> {
        let m = &self.0;
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
        if det == T::from(0) {
            return Err("Determinant is zero. There exists no inverse for this matrix!".into());
        }
        let inv_det = T::from(1) / det;

        Ok(Self::from_row_major([
            [t_a * inv_det, t_b * inv_det, t_c * inv_det],
            [t_d * inv_det, t_e * inv_det, t_f * inv_det],
            [t_g * inv_det, t_h * inv_det, t_i * inv_det],
        ]))
    }

    pub fn inverse(&self) -> Self {
        self.try_inverse().unwrap()
    }
}
