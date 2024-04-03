pub struct WState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WState {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                    limits: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        Self { device, queue }
    }
}
