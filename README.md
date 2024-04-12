# Rust warp

 - [ ] Make math implementations for vec types (i.e. dot, cross)
 - [ ] Make factory macros for vec types (i.e. wvec3(0, 1, 2) = underlying type with x, y, z set to 0, 1, 2)
 - [ ] Make macros for shorthanding the buffer generation, bind group, layouts, pipeline etc.
 - [x] Automatic value testing; write a macro that generates test cases for randomised structs, sending them to and from the gpu, checking if returned values are the same.
    - This would require defining the wgsl type within rust; does it make sense to therefore make tooling that includes this at the top of a wgsl file? I.e. reduce duplicate boiler
 - [x] Try encode 3channel image pixels in single u32 (resulted in a 4x speedup)