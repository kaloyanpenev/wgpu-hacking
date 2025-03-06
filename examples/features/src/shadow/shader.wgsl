struct Globals {
    view_proj: mat4x4<f32>,
    num_lights: vec4<u32>,
};

struct Vertex
{
    pos: vec4<f32>,
    normal: vec4<f32>
}

@group(0)
@binding(0)
var<uniform> u_globals: Globals;

@group(0)
@binding(4)
var<storage, read> u_spread_offsets: array<vec2<f32>>; 

@group(0)
@binding(5)
var<storage, read> verts_vs: array<Vertex>; // write the verts

struct Entity {
    world: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1)
@binding(0)
var<uniform> u_entity: Entity;

const grass_vertex_count = 128;

@vertex
fn vs_bake(@location(0) position: vec4<f32>,
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32
) -> @builtin(position) vec4<f32> {
    let spread_offset_mat = transpose(mat4x4f(
    // in x    y    z   1.0
        1.0,  0.0, 0.0, u_spread_offsets[instance_idx].x * saturate(f32(instance_idx)),
        0.0,  1.0, 0.0, 0.0,
        0.0,  0.0, 1.0, u_spread_offsets[instance_idx].y * saturate(f32(instance_idx)), 
        0.0,  0.0, 0.0, 1.0,
    ));
    let current_vert_idx = (instance_idx * grass_vertex_count) + vertex_idx;

    return u_globals.view_proj * u_entity.world * spread_offset_mat * vec4<f32>(verts_vs[current_vert_idx].pos.xyz, 1.0);
}

struct VertexOutput {
    @builtin(position) proj_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec4<f32>
};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32
) -> VertexOutput {
    let w = u_entity.world;
    let spread_offset_mat = transpose(mat4x4f(
        // in x    y    z   1.0
            1.0,  0.0, 0.0, u_spread_offsets[instance_idx].x * saturate(f32(instance_idx)),
            0.0,  1.0, 0.0, 0.0,
            0.0,  0.0, 1.0, u_spread_offsets[instance_idx].y * saturate(f32(instance_idx)), 
            0.0,  0.0, 0.0, 1.0,
        ));

    let current_vert_idx = (instance_idx * grass_vertex_count) + vertex_idx;
    let world_pos = u_entity.world * spread_offset_mat * vec4<f32>(verts_vs[current_vert_idx].pos.xyz, 1.0);
    //let world_pos = u_entity.world * spread_offset_mat * vec4<f32>(position);
    var result: VertexOutput;
    result.world_normal = mat3x3<f32>(w[0].xyz, w[1].xyz, w[2].xyz) * vec3<f32>(verts_vs[current_vert_idx].normal.xyz);
    //result.world_normal = mat3x3<f32>(w[0].xyz, w[1].xyz, w[2].xyz) * vec3<f32>(normal.xyz);
    result.world_position = world_pos;
    result.proj_position = u_globals.view_proj * world_pos;
    return result;
}

// fragment shader

struct Light {
    proj: mat4x4<f32>,
    pos: vec4<f32>,
    color: vec4<f32>,
};

@group(0)
@binding(1)
var<storage, read> s_lights: array<Light>;
@group(0)
@binding(1)
var<uniform> u_lights: array<Light, 10>; // Used when storage types are not supported
@group(0)
@binding(2)
var t_shadow: texture_depth_2d_array;
@group(0)
@binding(3)
var sampler_shadow: sampler_comparison;


fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
    if (homogeneous_coords.w <= 0.0) {
        return 1.0;
    }
    // compensate for the Y-flip difference between the NDC and texture coordinates
    let flip_correction = vec2<f32>(0.5, -0.5);
    // compute texture coordinates for shadow lookup
    let proj_correction = 1.0 / homogeneous_coords.w;
    let light_local = homogeneous_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    // do the lookup, using HW PCF and comparison
    return textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), homogeneous_coords.z * proj_correction);
}

const c_ambient: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const c_max_lights: u32 = 10u;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vertex.world_normal);
    // accumulate color
    var color: vec3<f32> = c_ambient;
    for(var i = 0u; i < min(u_globals.num_lights.x, c_max_lights); i += 1u) {
        let light = s_lights[i];
        // project into the light space
        let shadow = fetch_shadow(i, light.proj * vertex.world_position);
        // compute Lambertian diffuse term
        let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
        let diffuse = max(0.0, dot(normal, light_dir));
        // add light contribution
        color += diffuse * light.color.xyz * shadow * 2.0;
    }
    // multiply the light by material color
    return vec4<f32>(color, 1.0) * u_entity.color;
}

// The fragment entrypoint used when storage buffers are not available for the lights
@fragment
fn fs_main_without_storage(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vertex.world_normal);
    var color: vec3<f32> = c_ambient;
    for(var i = 0u; i < min(u_globals.num_lights.x, c_max_lights); i += 1u) {
        // This line is the only difference from the entrypoint above. It uses the lights
        // uniform instead of the lights storage buffer
        let light = u_lights[i];
        let shadow = fetch_shadow(i, light.proj * vertex.world_position);
        let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
        let diffuse = max(0.0, dot(normal, light_dir));
        color += shadow * diffuse * light.color.xyz;
    }
    return vec4<f32>(color, 1.0) * u_entity.color;
}


@group(0)
@binding(0)
var<storage, read_write> verts: array<Vertex>; // write the verts
@group(0)
@binding(1)
var<storage, read> v_entities: array<Entity>; // read the model so we can take into account where it is in the future - would need to compare pos to wind texture
@group(1)
@binding(0)
var<storage, read> wind_offsets: array<vec2<f32>>; 

@compute
@workgroup_size(64, 1, 1)
fn bezier_offset(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) wgid : vec3<u32>) {
    // based on https://www.desmos.com/calculator/d1ofwre0fr
    var wind = wind_offsets[wgid.x].x * 0.5;
    var p0 = vec2(0.0, 0.0);
    var p1 = vec2(0.0, 0.6);
    var p2 = vec2(0.0, 1.0 - wind);
    var p3 = vec2(1.25, 1.0);

    var t : f32 = f32(local_id.x) / 64.0;

    var bezier_eval_0x = ((1-t) * (1-t) * (1-t) * p0.x + t * p1.x);
    var bezier_eval_1x = t* ((1-t)*p1.x + t * p2.x);
    var bezier_eval_2x = t* ((1-t) * (1-t) * p1.x + t * p2.x);
    var bezier_eval_3x = t* ((1-t) * p2.x + t * p3.x);
    var bezier_x = bezier_eval_0x + bezier_eval_1x + bezier_eval_2x + bezier_eval_3x;

    var bezier_eval_0y = ((1-t) * (1-t) * p0.y + t * p1.y);
    var bezier_eval_1y = t* ((1-t)*p1.y + t * p2.y);
    var bezier_eval_2y = t* ((1-t) * (1-t) * p1.y + t * p2.y);
    var bezier_eval_3y = t* ((1-t) * p2.y + t * p3.y);
    var bezier_y = (1-t) * (bezier_eval_0y + bezier_eval_1y + bezier_eval_2y + bezier_eval_3y);

    // bezier x is towards normal (along z)
    // bezier y is up, along y
    var thid : u32 = global_id.x * 2; // stride is 2
    var vertex_offset : vec4<f32> = vec4(0.0, bezier_y, -bezier_x, 0.0); // assumes normal is Z but fix this
    verts[thid].pos = verts[thid].pos + vertex_offset;
    verts[thid + 1].pos = verts[thid + 1].pos + vertex_offset;
}
