use std::ops::{Mul, Sub};

use generic_array::ArrayLength;
use typenum::Prod;

// TODO: WScalars; AbstractInt, AbstractFloat, f16
pub trait WScalars {}
impl WScalars for bool {}
impl WScalars for u32 {}
impl WScalars for i32 {}
impl WScalars for f32 {}

/// Generics<Size in bytes, Alignment in bytes>
/// Padding in bytes = alignment - size
pub struct WPad<ALIGN, TYPESIZE, NUMTYPES>(
    generic_array::GenericArray<u8, typenum::op!(ALIGN - (TYPESIZE * NUMTYPES))>,
)
where
    TYPESIZE: Mul<NUMTYPES>,
    ALIGN: Sub<Prod<TYPESIZE, NUMTYPES>>,
    <ALIGN as Sub<Prod<TYPESIZE, NUMTYPES>>>::Output: ArrayLength;

#[macro_export]
macro_rules! size_of {
    ($type:ty) => {
        std::mem::size_of::<$type>()
    };
}

#[macro_export]
macro_rules! typenum {
    ($num:literal) => {
        paste::paste! {
            typenum::[<U $num>]
        }
    };
}

#[macro_export]
macro_rules! wvec_call {
    ($vec:ident, bool, $alignment:literal, $size:literal) => {
        $vec<bool, typenum!($alignment), typenum!(1)>
    };
    ($vec:ident, u32, $alignment:literal, $size:literal) => {
        $vec<u32, typenum!($alignment), typenum!(4)>
    };
    ($vec:ident, i32, $alignment:literal, $size:literal) => {
        $vec<i32, typenum!($alignment), typenum!(4)>
    };
    ($vec:ident, f32, $alignment:literal, $size:literal) => {
        $vec<f32, typenum!($alignment), typenum!(4)>
    };
}

macro_rules! define_vec {
    ($name:ident, $macro_name:ident, $n:literal, $fields:tt) => {
        pub struct $name<TYPE, ALIGN, TYPESIZE>
        where
            TYPE: WScalars,
            TYPESIZE: Mul<typenum!($n)>,
            ALIGN: Sub<Prod<TYPESIZE, typenum!($n)>>,
            <ALIGN as Sub<Prod<TYPESIZE, typenum!($n)>>>::Output: ArrayLength
            $fields

        #[macro_export]
        macro_rules! $macro_name {
            ($type:ident, $alignment:literal) => {
                wvec_call!($name, $type, $alignment, 2)
            };
        }
    };
}

define_vec!(WVec2, wvec2, 2, {
    pub x: TYPE,
    pub y: TYPE,
    _pad: WPad<ALIGN, TYPESIZE, typenum!(2)>,
});

define_vec!(WVec3, wvec3, 3, {
    pub x: TYPE,
    pub y: TYPE,
    pub z: TYPE,
    _pad: WPad<ALIGN, TYPESIZE, typenum!(3)>,
});

define_vec!(WVec4, wvec4, 4, {
    pub x: TYPE,
    pub y: TYPE,
    pub z: TYPE,
    pub w: TYPE,
    _pad: WPad<ALIGN, TYPESIZE, typenum!(4)>,
});
