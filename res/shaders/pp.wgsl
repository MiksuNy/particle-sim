enable wgpu_binding_array;

@group(0) @binding(0)
var cascade_textures: binding_array<texture_storage_2d<rgba32float, read_write>, 16>;

@group(0) @binding(1)
var pp_texture: texture_storage_2d<rgba16unorm, write>;

@compute @workgroup_size(8u, 8u, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var color = textureLoad(cascade_textures[0u], global_id.xy).rgb;
    color = aces_filmic(color);
    color = linear_to_srgb(color);

    textureStore(pp_texture, global_id.xy, vec4<f32>(color, 1.0f));
}

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(f32(linear.r < 0.0031308f), f32(linear.g < 0.0031308f), f32(linear.b < 0.0031308f));
    let higher = vec3<f32>(1.055) * pow(linear, vec3<f32>(1.0/2.4)) - vec3<f32>(0.055);
    let lower = linear * vec3<f32>(12.92);
    return mix(higher, lower, cutoff);
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51f;
    let b = 0.03f;
    let c = 2.43f;
    let d = 0.59f;
    let e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0f), vec3<f32>(1.0f));
}
