pub struct WState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WState {
    pub async fn new() -> Self {
        // let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            ..Default::default()
        });

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
                    required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        Self { device, queue }
    }
}

#[macro_export]
macro_rules! wbyte_cast {
    ($v:expr) => {
        bytemuck::cast_slice($v)
    };
}

#[macro_export]
macro_rules! wbyte_of {
    ($v:expr) => {
        bytemuck::bytes_of($v)
    };
}

#[macro_export]
macro_rules! wgpu_shader_load {
    ($dev:expr, $shader:expr, label=$label:expr) => {
        $dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: $label,
            source: wgpu::ShaderSource::Wgsl($shader.into()),
        })
    };
    ($dev:expr, $shader:expr) => {
        wgpu_shader_load!($dev, $shader, label = None)
    };
    ($name:expr, $dev:expr, $shader:expr) => {
        wgpu_shader_load!($dev, $shader, label = Some($name))
    };
}

#[macro_export]
macro_rules! wgpu_usage_literal {
    (MAP_READ) => {
        wgpu::BufferUsages::MAP_READ
    };
    (MAP_WRITE) => {
        wgpu::BufferUsages::MAP_WRITE
    };
    (COPY_SRC) => {
        wgpu::BufferUsages::COPY_SRC
    };
    (COPY_DST) => {
        wgpu::BufferUsages::COPY_DST
    };
    (INDEX) => {
        wgpu::BufferUsages::INDEX
    };
    (VERTEX) => {
        wgpu::BufferUsages::VERTEX
    };
    (UNIFORM) => {
        wgpu::BufferUsages::UNIFORM
    };
    (STORAGE) => {
        wgpu::BufferUsages::STORAGE
    };
    (INDIRECT) => {
        wgpu::BufferUsages::INDIRECT
    };
    (QUERY_RESOLVE) => {
        wgpu::BufferUsages::QUERY_RESOLVE
    };
}

#[macro_export]
macro_rules! wgpu_usages {
    ($u:ident) => {
        wgpu_usage_literal!($u)
    };
    ($u:ident | $($rest:tt)|*) => {
        wgpu_usage_literal!($u) | wgpu_usages!($($rest)|*)
    };
}

#[macro_export]
macro_rules! wgpu_buf_init {
    ($dev:expr, $bytes:expr, $usage:expr, label=$label:expr) => {
        $dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: $label,
            contents: $bytes,
            usage: $usage,
        })
    };

    ($dev:expr, $bytes:expr, [$($usages:tt)|*]) => {
        wgpu_buf_init!($dev, $bytes, wgpu_usages!($($usages)|*), label=None)
    };
    ($name:expr, $dev:expr, $bytes:expr, [$($usages:tt)|*]) => {
        wgpu_buf_init!($dev, $bytes, wgpu_usages!($($usages)|*), label=Some($name))
    };

    ($dev:expr, $bytes:expr, $usage:expr) => {
        wgpu_buf_init!($dev, $bytes, $usage, label=None)
    };
    ($name:expr, $dev:expr, $bytes:expr, $usage:expr) => {
        wgpu_buf_init!($dev, $bytes, $usage, label=Some($name))
    };
}

#[macro_export]
macro_rules! wgpu_buf {
    ($dev:expr, $size:expr, $usage:expr, $mapped:expr, label=$label:expr) => {
        $dev.create_buffer(&wgpu::BufferDescriptor {
            label: $label,
            size: $size,
            usage: $usage,
            mapped_at_creation: $mapped,
        })
    };

    ($dev:expr, $size:expr, [$($usages:tt)|*], $mapped:expr) => {
        wgpu_buf!($dev, $size, wgpu_usages!($($usages)|*), $mapped, label = None)
    };
    ($name:expr, $dev:expr, $size:expr, [$($usages:tt)|*], $mapped:expr) => {
        wgpu_buf!($dev, $size, wgpu_usages!($($usages)|*), $mapped, label = Some($name))
    };

    ($dev:expr, $size:expr, $usage:expr, $mapped:expr) => {
        wgpu_buf!($dev, $size, $usage, $mapped, label = None)
    };
    ($name:expr, $dev:expr, $size:expr, $usage:expr, $mapped:expr) => {
        wgpu_buf!($dev, $size, $usage, $mapped, label = Some($name))
    };
}

#[macro_export]
macro_rules! wgpu_bind_gle_compute {
    ($bind:expr, $read_only:expr) => {
        wgpu::BindGroupLayoutEntry {
            binding: $bind,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: $read_only,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
}

#[macro_export]
macro_rules! wgpu_bind_group_layout_compute {
    ($dev:expr, [$(($bind:expr, $read_only:expr)),*], label=$label:expr) => {
        $dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: $label,
            entries: &[
                $(wgpu_bind_gle_compute!($bind, $read_only)),*
            ],
        })
    };

    ($dev:expr, [$(($bind:expr, $read_only:expr)),*]) => {
        wgpu_bind_group_layout_compute!($dev, [$(($bind, $read_only)),*], label=None)
    };
    ($name:expr, $dev:expr, [$(($bind:expr, $read_only:expr)),*]) => {
        wgpu_bind_group_layout_compute!($dev, [$(($bind, $read_only)),*], label=Some($name))
    };
}

#[macro_export]
macro_rules! wgpu_compute_pipeline_layout {
    ($dev:expr, $bind:expr, label=$label:expr) => {
        $dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: $label,
            bind_group_layouts: $bind,
            push_constant_ranges: &[],
        })
    };

    ($dev:expr, $bind:expr) => {
        wgpu_compute_pipeline_layout!($dev, $bind, label = None)
    };
    ($name:expr, $dev:expr, $bind:expr, $push:expr) => {
        wgpu_compute_pipeline_layout!($dev, $bind, label = Some($name))
    };
}

#[macro_export]
macro_rules! wgpu_compute_pipeline {
    ($dev:expr, $layout:expr, $module:expr, $entry:expr, label=$label:expr) => {
        $dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: $label,
            layout: Some($layout),
            module: $module,
            entry_point: $entry,
        })
    };

    ($dev:expr, $layout:expr, $module:expr, $entry:expr) => {
        wgpu_compute_pipeline!($dev, $layout, $module, $entry, label = None)
    };
    ($name:expr, $dev:expr, $layout:expr, $module:expr, $entry:expr) => {
        wgpu_compute_pipeline!($dev, $layout, $module, $entry, label = Some($name))
    };
}

#[macro_export]
macro_rules! wgpu_bind_group {
    ($dev:expr, $layout:expr, [$(($bind:expr, $resource:expr)),*], label=$label:expr) => {
        $dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: $label,
            layout: $layout,
            entries: &[
                $(wgpu::BindGroupEntry {
                    binding: $bind,
                    resource: $resource,
                }),*
            ],
        })
    };

    ($dev:expr, $layout:expr, [$(($bind:expr, $resource:expr)),*]) => {
        wgpu_bind_group!($dev, $layout, [$(($bind, $resource)),*], label=None)
    };
    ($name:expr, $dev:expr, $layout:expr, [$(($bind:expr, $resource:expr)),*]) => {
        wgpu_bind_group!($dev, $layout, [$(($bind, $resource)),*], label=Some($name))
    };
}

pub struct WTsQueryState {
    pub max_count: u32,
    pub current: u32,
    pub set: wgpu::QuerySet,
    pub buf: wgpu::Buffer,
    pub out: wgpu::Buffer,
}

impl WTsQueryState {
    pub fn new(dev: &wgpu::Device, c: u32) -> Self {
        WTsQueryState {
            max_count: c,
            current: 0,
            set: dev.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                count: c,
                ty: wgpu::QueryType::Timestamp,
            }),
            buf: wgpu_buf!(
                dev,
                c as u64 * 8,
                [QUERY_RESOLVE | STORAGE | COPY_SRC | COPY_DST],
                false
            ),
            out: wgpu_buf!(dev, 16, [MAP_READ | COPY_DST], false),
        }
    }

    pub fn write(&mut self, encoder: &mut wgpu::CommandEncoder) {
        assert!(
            self.current < self.max_count,
            "Timestamp query count exceeded"
        );
        encoder.write_timestamp(&self.set, self.current);
        self.current += 1;
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.set, 0..self.max_count, &self.buf, 0);
        encoder.copy_buffer_to_buffer(&self.buf, 0, &self.out, 0, self.max_count as u64 * 8);
    }

    pub fn map_async(&mut self) -> (usize, wgpu::BufferSlice) {
        let query_slice = self.out.slice(..);
        query_slice.map_async(wgpu::MapMode::Read, move |_| ());
        (self.max_count as usize, query_slice)
    }

    pub fn read(count: usize, slice: wgpu::BufferSlice) -> Vec<u64> {
        let data = slice.get_mapped_range();
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let start = i as usize * 8;
            let end = start + 8;
            let bytes = &data[start..end];
            let value = u64::from_ne_bytes(bytes.try_into().unwrap());
            result.push(value);
        }
        result
    }
}
