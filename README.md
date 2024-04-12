# Rust warp

 - [ ] Make math implementations for vec types (i.e. dot, cross)
 - [ ] Make factory macros for vec types (i.e. wvec3(0, 1, 2) = underlying type with x, y, z set to 0, 1, 2)
 - [ ] Make macros for shorthanding the buffer generation, bind group, layouts, pipeline etc.
 - [x] Automatic value testing; write a macro that generates test cases for randomised structs, sending them to and from the gpu, checking if returned values are the same.
    - This would require defining the wgsl type within rust; does it make sense to therefore make tooling that includes this at the top of a wgsl file? I.e. reduce duplicate boiler
 - [x] Try encode 3channel image pixels in single u32 (resulted in a 4x speedup)

### Needleman Wunsch GPU sync idea
 - Using synchronisation method from data_dep.wgsl
 - Max workgroup size = 256 = 16x16
 - Needleman wunsch has diagonal dependencies (i.e. up, left, up+left)
 - Divide alignment problem into 16x16 workgroups which are solved diagonally. Diagonal iters = (((n+1)/16) + ((m+1)/16)) + 1
   - Inside those 16x16 groups, same principal is applied, however, each workgroup has shared memory.
     - This means the i=0 and j=0 rows and columns take their values from global memory
     - i=15 and j=15 rows save their values in global memory (edge case where final column/row workgroup and sequence is not a multiple of 16)
     - Here, there would be 16 + 16 + 1 workgroup memory synchronisations happening.

