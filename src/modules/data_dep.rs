use wgpu::util::DeviceExt;

use crate::setup::*;

pub async fn data_dep_gpu(state: &mut WState) {
    let dev = &state.device;

    let cs_module = wgpu_shader_load!(dev, include_str!("data_dep.wgsl"));
    let mut query_state = WTsQueryState::new(dev, 2);

    const N: usize = 64 * 4 * 2;
    let in_sq = vec![0i32; N];
    let in_bytes: &[u8] = wbyte_cast!(&in_sq);

    let in_buffer = wgpu_buf_init!(dev, in_bytes, [STORAGE | COPY_SRC | COPY_DST]);
    let out_buf = wgpu_buf!(dev, in_bytes.len() as u64, [MAP_READ | COPY_DST], false);

    let bind_group_layout = wgpu_bind_group_layout_compute!(dev, [(0, true), (1, false)]);
    let compute_pipeline_layout = wgpu_compute_pipeline_layout!(dev, &[&bind_group_layout]);
    let pipeline = wgpu_compute_pipeline!(dev, &compute_pipeline_layout, &cs_module, "main");

    let i = 0;
    let pass_bytes = wbyte_of!(&i);
    let pass_buffer = wgpu_buf_init!(dev, pass_bytes, [STORAGE | COPY_SRC | COPY_DST]);

    let bind_group = wgpu_bind_group!(
        dev,
        &bind_group_layout,
        [
            (0, pass_buffer.as_entire_binding()),
            (1, in_buffer.as_entire_binding())
        ]
    );

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
    encoder.copy_buffer_to_buffer(&in_buffer, 0, &out_buf, 0, in_bytes.len() as u64);
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
    // println!("Elapsed: {:?}", start.elapsed());
}
