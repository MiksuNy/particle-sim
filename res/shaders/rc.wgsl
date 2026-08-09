enable wgpu_binding_array;

const PI = 3.1415926535f;
const BASE_INTERVAL_LENGTH = 0.5f;

@group(0) @binding(0)
var cascade_textures: binding_array<texture_storage_2d<rgba32float, read_write>, 16>;

var<immediate> immediates: Immediates;

struct Immediates {
    cascade_index: u32,
    cursor_x: u32,
    cursor_y: u32
}

@compute @workgroup_size(8u, 8u, 1u)
fn gen_cascades(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let probe_size = probe_size(immediates.cascade_index);
    let probe_coord = global_id.xy % probe_size;
    let probe_center = vec2<f32>(global_id.xy - probe_coord) + vec2<f32>(probe_size) * 0.5f;

    let ray_index = probe_coord.y * probe_size.x + probe_coord.x;
    let ray_count = probe_size.x * probe_size.y;
    let ray_angle = 2.0f * PI * ((f32(ray_index) + 0.5f) / f32(ray_count));
    let ray_dir = vec2<f32>(cos(ray_angle), sin(ray_angle));

    let range = interval_range(immediates.cascade_index);
    let start = probe_center + ray_dir * range.x;
    let end = probe_center + ray_dir * range.y;
    let radiance = cast_interval(start, end);

    let debug_color = vec4<f32>(
        (radiance.r * 0.5f) + ((f32(probe_coord.x) / f32(probe_size.x)) * 0.5f),
        (radiance.g * 0.5f) + ((f32(probe_coord.y) / f32(probe_size.y)) * 0.5f),
        (radiance.b * 0.5f),
        1.0f
    );

    textureStore(cascade_textures[immediates.cascade_index], global_id.xy, radiance);
}

// Merge all cascades from top (lowest spatial res) to down (highest spatial res)
@compute @workgroup_size(8u, 8u, 1u)
fn merge_cascades(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dest_size = probe_size(immediates.cascade_index);
    let dest_coord = global_id.xy % dest_size;
    let dest_center = vec2<f32>(global_id.xy - dest_coord) + vec2<f32>(dest_size) * 0.5f;

    let dir_index = dest_coord.y * dest_size.x + dest_coord.x;

    let bilinear_size = vec2<f32>(probe_size(immediates.cascade_index + 1u));
    var weights: vec4<f32>;
    var base_index: vec2<u32>;
    bilinear_samples(dest_center, bilinear_size, &weights, &base_index);

    let dest_interval = textureLoad(cascade_textures[immediates.cascade_index], global_id.xy);

    var merged = vec4<f32>(0.0f);
    for (var d = 0u; d < 4u; d++) {
        var radiance = vec4<f32>(0.0f);
        for (var b = 0u; b < 4u; b++) {
            let base_offset = bilinear_offset(b);
            let bilinear_index = base_index + base_offset;
            let base_dir_index = dir_index * 4u;
            let bilinear_dir_index = base_dir_index + d;
            let bilinear_dir_coord = vec2<u32>(
                u32(f32(bilinear_dir_index) % bilinear_size.x),
                u32(f32(bilinear_dir_index) / bilinear_size.x)
            );
            let bilinear_texel = vec2<u32>(vec2<f32>(bilinear_index) * bilinear_size + vec2<f32>(bilinear_dir_coord));
            let bilinear_interval = textureLoad(cascade_textures[immediates.cascade_index + 1u], bilinear_texel);

            radiance += merge_intervals(dest_interval, bilinear_interval) * weights[b];
        }
        merged += radiance / 4.0f;
    }

    textureStore(cascade_textures[immediates.cascade_index], global_id.xy, merged);
}

// Bilinearly interpolate 4 nearest cascade 0 probes
@compute @workgroup_size(8u, 8u, 1u)
fn final_pass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dest_size = vec2<u32>(1u, 1u);
    let dest_coord = global_id.xy % dest_size;
    let dest_center = vec2<f32>(global_id.xy - dest_coord) + vec2<f32>(dest_size) * 0.5f;

    let dir_index = dest_coord.y * dest_size.x + dest_coord.x;

    let bilinear_size = vec2<f32>(dest_size);
    var weights: vec4<f32>;
    var base_index: vec2<u32>;
    bilinear_samples(dest_center, bilinear_size, &weights, &base_index);

    let dest_interval = textureLoad(cascade_textures[0u], global_id.xy);

    var merged = vec4<f32>(0.0f);
    for (var d = 0u; d < 4u; d++) {
        var radiance = vec4<f32>(0.0f);
        for (var b = 0u; b < 4u; b++) {
            let base_offset = bilinear_offset(b);
            let bilinear_index = base_index + base_offset;
            let base_dir_index = dir_index;
            let bilinear_dir_index = base_dir_index + d;
            let bilinear_dir_coord = vec2<u32>(
                u32(f32(bilinear_dir_index) % bilinear_size.x),
                u32(f32(bilinear_dir_index) / bilinear_size.x)
            );
            let bilinear_texel = vec2<u32>(vec2<f32>(bilinear_index) * bilinear_size + vec2<f32>(bilinear_dir_coord));
            let bilinear_interval = textureLoad(cascade_textures[0u], bilinear_texel);

            radiance += merge_intervals(dest_interval, bilinear_interval) * weights[b];
        }
        merged += radiance / 4.0f;
    }

    textureStore(cascade_textures[0u], global_id.xy, merged);
}

fn bilinear_weights(ratio: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(
        (1.0f - ratio.x) * (1.0f - ratio.y),
        ratio.x * (1.0f - ratio.y),
        (1.0f - ratio.x) * ratio.y,
        ratio.x * ratio.y
    );
}

fn bilinear_samples(dest_center: vec2<f32>, bilinear_size: vec2<f32>, weights: ptr<function, vec4<f32>>, base_index: ptr<function, vec2<u32>>) {
    let base_coord = (dest_center / bilinear_size) - vec2<f32>(0.5f, 0.5f);
    let ratio = fract(base_coord);
    *weights = bilinear_weights(ratio);
    *base_index = vec2<u32>(floor(base_coord));
}

fn bilinear_offset(offset_index: u32) -> vec2<u32> {
    let offsets = array<vec2<u32>, 4u>(
        vec2<u32>(0u, 0u),
        vec2<u32>(1u, 0u),
        vec2<u32>(0u, 1u),
        vec2<u32>(1u, 1u)
    );
    return offsets[offset_index];
}

fn cast_interval(interval_start: vec2<f32>, interval_end: vec2<f32>) -> vec4<f32> {
    const NUM_OBJECTS: u32 = 5u;
    let objects = array<vec3<f32>, NUM_OBJECTS>(
        vec3<f32>(320.0f, 320.0f, 30.0f),
        vec3<f32>(160.0f, 320.0f, 30.0f),
        vec3<f32>(320.0f, 160.0f, 30.0f),
        vec3<f32>(160.0f, 160.0f, 30.0f),
        vec3<f32>(f32(immediates.cursor_x), f32(immediates.cursor_y), 10.0f)
    );
    let colors = array<vec4<f32>, NUM_OBJECTS>(
        vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f),
        vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f),
        vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f),
        vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f),
        vec4<f32>(0.1f, 0.1f, 0.1f, 0.0f)
    );

    let t_max = distance(interval_start, interval_end);
    let ray_dir = normalize(interval_end - interval_start);

    var radiance = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);

    var hit_object_index = 0u;
    var t = 0.0f;
    for (var i = 0u; i < 64u; i++) {
        var d = 1e30f;
        for (var object_index = 0u; object_index < NUM_OBJECTS; object_index++) {
            let temp_d = sd_sphere(interval_start + ray_dir * t, objects[object_index].xy, objects[object_index].z);
            if temp_d < d {
                hit_object_index = object_index;
                d = temp_d;
            }
        }

        t += abs(d);

        if t >= t_max {
            break;
        }

        if 0.1f < t && d < 1.0f {
            radiance.r += colors[hit_object_index].r;
            radiance.g += colors[hit_object_index].g;
            radiance.b += colors[hit_object_index].b;

            radiance.a *= colors[hit_object_index].a;
            break;
        }
    }

    return radiance;
}

fn probe_size(cascade_index: u32) -> vec2<u32> {
    return vec2<u32>(1u << cascade_index, 1u << cascade_index);
}

fn interval_range(cascade_index: u32) -> vec2<f32> {
    let start = (BASE_INTERVAL_LENGTH * (1.0f - pow(4.0f, f32(cascade_index)))) / (1.0f - 4.0f);
    var end = BASE_INTERVAL_LENGTH * pow(4.0f, f32(cascade_index));

    let d = pow(2.0f, f32(cascade_index + 1u));
    end += f32(cascade_index) * length(vec2<f32>(d, d));

    return vec2<f32>(start, end);
}

fn merge_intervals(near: vec4<f32>, far: vec4<f32>) -> vec4<f32> {
    let radiance = near.rgb + (far.rgb * near.a);
    return vec4<f32>(radiance, near.a * far.a);
}

fn sd_sphere(p: vec2<f32>, c: vec2<f32>, r: f32) -> f32 {
    return distance(p, c) - r;
}
