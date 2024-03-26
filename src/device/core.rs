use generic_array::ArrayLength;
use std::ops::{Mul, Sub};
use typenum::{Diff, Prod};

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

/// Generics<Size in bytes, Alignment in bytes>
/// Padding in bytes = alignment - size
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct WPad<ALIGN, TYPESIZE, NUMTYPES>(
    generic_array::GenericArray<i8, Diff<ALIGN, Prod<TYPESIZE, NUMTYPES>>>,
)
where
    TYPESIZE: Mul<NUMTYPES>,
    ALIGN: Sub<Prod<TYPESIZE, NUMTYPES>>,
    <ALIGN as Sub<Prod<TYPESIZE, NUMTYPES>>>::Output: ArrayLength,
    <<ALIGN as Sub<<TYPESIZE as Mul<NUMTYPES>>::Output>>::Output as ArrayLength>::ArrayType<i8>:
        Copy;

// TODO: Is implementing this trait ok?
unsafe impl<ALIGN, TYPESIZE, NUMTYPES> bytemuck::Zeroable for WPad<ALIGN, TYPESIZE, NUMTYPES>
where
    TYPESIZE: Mul<NUMTYPES>,
    ALIGN: Sub<Prod<TYPESIZE, NUMTYPES>>,
    <ALIGN as Sub<Prod<TYPESIZE, NUMTYPES>>>::Output: ArrayLength,
    <<ALIGN as Sub<<TYPESIZE as Mul<NUMTYPES>>::Output>>::Output as ArrayLength>::ArrayType<i8>:
        Copy,
{
    fn zeroed() -> Self {
        Self(generic_array::GenericArray::default())
    }
}

// TODO: Is implementing this one ok???
unsafe impl<ALIGN, TYPESIZE, NUMTYPES> bytemuck::Pod for WPad<ALIGN, TYPESIZE, NUMTYPES>
where
    TYPESIZE: Mul<NUMTYPES> + Copy + 'static,
    ALIGN: Sub<Prod<TYPESIZE, NUMTYPES>> + Copy + 'static,
    <ALIGN as Sub<Prod<TYPESIZE, NUMTYPES>>>::Output: ArrayLength,
    <<ALIGN as Sub<<TYPESIZE as Mul<NUMTYPES>>::Output>>::Output as ArrayLength>::ArrayType<i8>:
        Copy,
    NUMTYPES: Copy + 'static,
{
}

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
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Default, bytemuck::Zeroable)]
        pub struct $name<TYPE, ALIGN, TYPESIZE>
        where
            TYPE: WScalars,
            TYPESIZE: Mul<typenum!($n)>,
            ALIGN: Sub<Prod<TYPESIZE, typenum!($n)>>,
            <ALIGN as Sub<Prod<TYPESIZE, typenum!($n)>>>::Output: ArrayLength,
            <<ALIGN as std::ops::Sub<<TYPESIZE as Mul<typenum!($n)>>::Output>>::Output as ArrayLength>::ArrayType<i8>: std::marker::Copy
            $fields

        unsafe impl <TYPE, ALIGN, TYPESIZE> bytemuck::Pod for $name<TYPE, ALIGN, TYPESIZE>
        where
            TYPE: WScalars,
            TYPESIZE: Mul<typenum!($n)>,
            ALIGN: Sub<Prod<TYPESIZE, typenum!($n)>>,
            <ALIGN as Sub<Prod<TYPESIZE, typenum!($n)>>>::Output: ArrayLength,
            <<ALIGN as Sub<<TYPESIZE as Mul<typenum!($n)>>::Output>>::Output as ArrayLength>::ArrayType<i8>: Copy,
            TYPESIZE: bytemuck::Zeroable + Copy + 'static,
            ALIGN: bytemuck::Zeroable + Copy + 'static,
            TYPE: bytemuck::Zeroable + Copy + 'static
        {}

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
    _pad: WPad<ALIGN, TYPESIZE, typenum::U2>,
});

// define_vec!(WVec3, wvec3, 3, {
//     pub x: TYPE,
//     pub y: TYPE,
//     pub z: TYPE,
//     _pad: WPad<ALIGN, TYPESIZE, typenum::U3>,
// });

// define_vec!(WVec4, wvec4, 4, {
//     pub x: TYPE,
//     pub y: TYPE,
//     pub z: TYPE,
//     pub w: TYPE,
//     _pad: WPad<ALIGN, TYPESIZE, typenum::U4>,
// });

struct Matrix<T, const ROWS: usize, const COLS: usize, const COLALIGNMENT: usize>
where
    T: Default + Copy,
{
    data: [[T; COLALIGNMENT]; ROWS],
}

impl<T: Default + Copy, const ROWS: usize, const COLS: usize, const COLALIGNMENT: usize>
    Matrix<T, ROWS, COLS, COLALIGNMENT>
{
    pub fn new() -> Self {
        Self {
            data: [[T::default(); COLALIGNMENT]; ROWS],
        }
    }

    pub fn set(&mut self, row: usize, col: usize, val: T) {
        assert!(row < ROWS && col < COLS, "Index out of bounds");
        self.data[row][col] = val;
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(row < ROWS && col < COLS, "Index out of bounds");
        self.data[row][col]
    }
}
