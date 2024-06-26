# Rust warp

This is a project that I used to learn about:
 - GPU programming with [WGSL](https://www.w3.org/TR/WGSL/), aswell as writing Rust macros
 - Writing [Rust macros](https://github.com/Sean-McConnachie/rustwarp/blob/main/src/types.rs)

## What's in this project?
 - Vector structs generated by macros with different compile-time sizes for appropriate padding `wvec3!(u32, 4)` = a vector3 type with 4 bytes of padding.
 - Automatic test generation for types that implement the `WTestable` trait. When calling your `Type` with `wtest!(Type, 1049)`, for example, it will generate 1049 instances of `Type` with random variables, copy these into the GPU, then copy them back to ensure there are no mis-alignment issues.
 - Another project I was working on did a lot of image transformations using `OpenCL` in `OpenCV`; copying the image took from host to GPU so much time that the performance benefit gained from a massively parrellel GPU was simply negated. A solution I came up with was to encode a 3-channel 8-bit image into a single 32-bit integer - BOOM! 4x speedup xd (likely 4x due to padding on a 3-channel type)
 - I naively tried to implement dynamic programming on a GPU and had some success, but nothing that would beat a normal CPU and certainly nothing that would beat the simplicity of writing DP on CPU.

### Todo:
 - [ ] Add examples
 - [ ] Refactor `multiple_warp.rs` and `warp_perspective.rs` to use macros
 - [ ] Make math implementations for vec types (i.e. dot, cross)
 - [x] Automatic value testing; write a macro that generates test cases for randomised structs, sending them to and from the gpu, checking if returned values are the same.
    - This would require defining the wgsl type within rust; does it make sense to therefore make tooling that includes this at the top of a wgsl file? I.e. reduce duplicate boiler
 - [x] Try encode 3channel image pixels in single u32 (resulted in a 4x speedup)
