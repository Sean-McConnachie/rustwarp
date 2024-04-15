use wgpu::util::DeviceExt;

use crate::setup::*;

pub async fn invocations_gpu(state: &mut WState) {
    let dev = &state.device;

    const WG_ROWS: u32 = 2;
    const WG_COLS: u32 = 2;
    let shader = wstring_replace!(
        include_str!("invocations.wgsl"),
        [
            ("{{WG_ROWS}}", &WG_ROWS.to_string()),
            ("{{WG_COLS}}", &WG_COLS.to_string())
        ]
    );
    println!("{shader}");
    let cs_module = wgpu_shader_load!(dev, shader);
    let mut query_state = WTsQueryState::new(dev, 2);

    const N: usize = (WG_COLS * WG_ROWS) as usize * 2;
    let in_seq = vec![-1; N];

    let in_bytes: &[u8] = wbyte_cast!(&in_seq);

    let in_buf = wgpu_buf_init!(dev, in_bytes, [STORAGE | COPY_SRC | COPY_DST]);
    let out_buf = wgpu_buf!(dev, in_bytes.len() as u64, [MAP_READ | COPY_DST], false);

    let bind_group_layout = wgpu_bind_group_layout_compute!(dev, [(0, false)]);
    let compute_pipeline_layout = wgpu_compute_pipeline_layout!(dev, &[&bind_group_layout]);
    let pipeline = wgpu_compute_pipeline!(dev, &compute_pipeline_layout, &cs_module, "main");

    let bind_group = wgpu_bind_group!(dev, &bind_group_layout, [(0, in_buf.as_entire_binding())]);

    let mut encoder = state.device.create_command_encoder(&Default::default());
    query_state.write(&mut encoder);
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(N as u32, 1, 1);
    }
    query_state.write(&mut encoder);
    state.queue.submit(Some(encoder.finish()));

    let mut encoder = state.device.create_command_encoder(&Default::default());
    // Get data out of device
    encoder.copy_buffer_to_buffer(&in_buf, 0, &out_buf, 0, in_bytes.len() as u64);
    query_state.resolve(&mut encoder);
    state.queue.submit(Some(encoder.finish()));

    let out_slice = out_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    out_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // More queries
    let (query_count, query_slice) = query_state.map_async();
    state.device.poll(wgpu::Maintain::Wait);

    let ts_data = WTsQueryState::read(query_count, query_slice);
    let ts_period = state.queue.get_timestamp_period();
    println!(
        "compute shader elapsed: {:?}ms",
        (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
    );

    if let Some(Ok(())) = receiver.receive().await {
        let data_raw = &*out_slice.get_mapped_range();
        let data: &[i32] = bytemuck::cast_slice(data_raw);
        println!("{:?}", data);
    }
}
