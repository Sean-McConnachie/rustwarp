use bytemuck::{Pod, Zeroable};
use paste::paste;

// TODO: WScalars; AbstractInt, AbstractFloat, f16
pub trait WScalars {}
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

macro_rules! wvec_def_struct {
    ($struct:ident { $($fields:ident),* }, $pad:expr) => {
        paste! {
            #[repr(C)]
            #[derive(Copy, Clone, Debug, Default, Zeroable)]
            pub struct [< $struct P $pad >] <T>
            where
                T: WScalars + Copy,
            {
                $(pub $fields: T,)*
                _pad: [u8; $pad],
            }

            unsafe impl<T> Pod for [< $struct P $pad >]<T>
            where
                T: WScalars + Copy + Zeroable + 'static,
            { }
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
                        crate::device::core::[< $struct P $pad >]<$typ>
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
