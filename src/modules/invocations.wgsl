
@group(0)
@binding(0)
var<storage, read_write> arr: array<i32>;

const WG_COLS: u32 = {{WG_COLS}};
const WG_ROWS: u32 = {{WG_ROWS}};
const WG_SIZE: u32 = WG_COLS * WG_ROWS;

@compute
@workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lcl_y = lid.x / WG_COLS;
    let lcl_x = lid.x % WG_COLS;
    let lcl_i = (lcl_y * WG_COLS) + lcl_x;

    let gbl_i = gid.x / WG_SIZE;
    arr[gid.x] = i32(wid.x);
}