import numpy as np
import warp as wp

import omni.usd
import omni.timeline
from usdrt import Usd, UsdGeom, Vt, Sdf, Gf


################################################################################
####################  START OF NVIDIA OCEAN DEFORM EXAMPLE  ####################
################################################################################

PROFILE_EXTENT = 410.0
PROFILE_RES = 8192
PROFILE_WAVENUM = 1000
MIN_WAVE_LENGTH = 0.1
MAX_WAVE_LENGTH = 250.0


#   Kernels
# ------------------------------------------------------------------------------


# fractional part of a (w.r.t. floor(a))
@wp.func
def frac(a: float):
    return a - wp.floor(a)


# square of a
@wp.func
def sqr(a: float):
    return a * a


@wp.func
def alpha_beta_spectrum(omega: float, peak_omega: float, alpha: float, beta: float, gravity: float):
    return (alpha * gravity * gravity / wp.pow(omega, 5.0)) * wp.exp(-beta * wp.pow(peak_omega / omega, 4.0))


@wp.func
def jonswap_peak_sharpening(omega: float, peak_omega: float, gamma: float):
    sigma = float(0.07)
    if omega > peak_omega:
        sigma = float(0.09)
    return wp.pow(gamma, wp.exp(-0.5 * sqr((omega - peak_omega) / (sigma * peak_omega))))


@wp.func
def jonswap_spectrum(omega: float, gravity: float, alpha: float, peak_omega: float, gamma: float):
    # https://www.wikiwaves.org/index.php/Ocean-Wave_Spectra#JONSWAP_Spectrum
    # CHANGE: use computed alpha and peak_omega from update_profile
    return jonswap_peak_sharpening(omega, peak_omega, gamma) * alpha_beta_spectrum(
        omega, peak_omega, alpha, 1.25, gravity
    )


@wp.func
def TMA_spectrum(omega: float, gravity: float, alpha: float, peak_omega: float, gamma: float, water_depth: float):
    # https://dl.acm.org/doi/10.1145/2791261.2791267
    omegaH = omega * wp.sqrt(water_depth / gravity)
    # Clipped at 2.0, not 2.2: the branch below (1 - 0.5*(2-omegaH)^2)
    # already equals exactly 1.0 at omegaH=2.0, matching the standard
    # Kitaigorodskii TMA correction's constant Phi=1 branch for omegaH>=2.
    omegaH = wp.max(0.0, wp.min(2.0, omegaH))
    phi = 0.5 * omegaH * omegaH
    if omegaH > 1.0:
        phi = 1.0 - 0.5 * sqr(2.0 - omegaH)
    return phi * jonswap_spectrum(omega, gravity, alpha, peak_omega, gamma)


# warp kernel definitions
@wp.kernel
def update_profile(
    profile: wp.array(dtype=wp.vec3),
    profile_res: int,
    profile_data_num: int,
    min_lambda: float,
    max_lambda: float,
    profile_extend: float,
    time: float,
    wind_speed: float,
    fetch_km: float,
    gamma: float,
    water_depth: float,
):
    x = wp.tid()
    randState = wp.rand_init(7)
    # sampling parameters
    omega0 = wp.sqrt(wp.tau * 9.80665 / min_lambda)
    omega1 = wp.sqrt(wp.tau * 9.80665 / max_lambda)
    omega_delta = wp.abs(omega1 - omega0) / float(profile_data_num)
    # we blend three displacements for seamless spatial profile tiling
    space_pos_1 = profile_extend * float(x) / float(profile_res)
    space_pos_2 = space_pos_1 + profile_extend
    space_pos_3 = space_pos_1 - profile_extend
    p1 = wp.vec2(0.0, 0.0)
    p2 = wp.vec2(0.0, 0.0)
    p3 = wp.vec2(0.0, 0.0)
    # CHANGE: compute alpha and peak_omega once per profile update
    fetch = 1000.0 * fetch_km
    alpha = 0.076 * wp.pow(wind_speed * wind_speed / (9.80665 * fetch), 0.22)
    peak_omega = 22.0 * wp.pow(wp.abs(9.80665 * 9.80665 / (wind_speed * fetch)), 1.0 / 3.0)
    for i in range(profile_data_num):
        omega = wp.abs(omega0 + (omega1 - omega0) * float(i) / float(profile_data_num))  # linear sampling of omega
        k = omega * omega / 9.80665
        phase = -time * omega + wp.randf(randState) * 2.0 * wp.pi
        amplitude = float(10000.0) * wp.sqrt(
            wp.abs(2.0 * omega_delta * TMA_spectrum(omega, 9.80665, alpha, peak_omega, gamma, water_depth))
        )
        p1 = wp.vec2(
            p1[0] + amplitude * wp.sin(phase + space_pos_1 * k), p1[1] - amplitude * wp.cos(phase + space_pos_1 * k)
        )
        p2 = wp.vec2(
            p2[0] + amplitude * wp.sin(phase + space_pos_2 * k), p2[1] - amplitude * wp.cos(phase + space_pos_2 * k)
        )
        p3 = wp.vec2(
            p3[0] + amplitude * wp.sin(phase + space_pos_3 * k), p3[1] - amplitude * wp.cos(phase + space_pos_3 * k)
        )
    # cubic blending coefficients
    s = float(float(x) / float(profile_res))
    c1 = float(2.0 * s * s * s - 3.0 * s * s + 1.0)
    c2 = float(-2.0 * s * s * s + 3.0 * s * s)
    disp_out = wp.vec3(
        (p1[0] + c1 * p2[0] + c2 * p3[0]) / float(profile_data_num),
        (p1[1] + c1 * p2[1] + c2 * p3[1]) / float(profile_data_num),
        0.0,
    )
    profile[x] = disp_out


@wp.kernel
def update_points(
    points: wp.array(dtype=wp.vec3),
    profile: wp.array(dtype=wp.vec3),
    profile_res: int,
    profile_extent: float,
    amplitude: float,
    center_pos: wp.vec3,
    clipmap_cell_size: float,
    direction_data: wp.array(dtype=wp.vec3), 
    direction_count: int,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p_crd = wp.vec3(
        points[tid][0] + wp.floor(center_pos[0] / clipmap_cell_size) * clipmap_cell_size,
        points[tid][1],
        points[tid][2] + wp.floor(center_pos[2] / clipmap_cell_size) * clipmap_cell_size,
    )

    randState = wp.rand_init(7)
    disp_x = float(0.0)
    disp_y = float(0.0)
    disp_z = float(0.0)
    w_sum = float(0.0)
    for d in range(0, direction_count):
        dir_vec = direction_data[d] 
        dir_x = dir_vec[0]
        dir_y = dir_vec[1]
        dir_amp = dir_vec[2]
        rand_phase = wp.randf(randState)
        x_crd = (p_crd[0] * dir_x + p_crd[1] * dir_y) / profile_extent + rand_phase 
        pos_0 = int(wp.floor(x_crd * float(profile_res))) % profile_res
        if x_crd < 0.0:
            pos_0 = pos_0 + profile_res - 1
        pos_1 = int(pos_0 + 1) % profile_res
        p_disp_0 = profile[pos_0]
        p_disp_1 = profile[pos_1]
        w = frac(x_crd * float(profile_res))
        prof_height_x = dir_amp * float((1.0 - w) * p_disp_0[0] + w * p_disp_1[0])
        prof_height_y = dir_amp * float((1.0 - w) * p_disp_0[1] + w * p_disp_1[1])
        disp_x = disp_x + dir_x * prof_height_x
        disp_y = disp_y + dir_y * prof_height_x 
        disp_z = disp_z + prof_height_y  
        w_sum = w_sum + 1.0

    # write output vertex position
    out_points[tid] = wp.vec3(
        p_crd[0] + amplitude * disp_x / w_sum,
        p_crd[1] + amplitude * disp_y / w_sum,
        p_crd[2] + amplitude * disp_z / w_sum,
    )


################################################################################
####################   END OF NVIDIA OCEAN DEFORM EXAMPLE   ####################
################################################################################


@wp.kernel
def precompute_directions(
    direction_data: wp.array(dtype=wp.vec3),
    total_amp: wp.array(dtype=float),
    directionality: float,
    direction: float,
    direction_count: int
):
    d = wp.tid()
    r = float(d) * wp.tau / float(direction_count) + 0.02
    dir_x = wp.cos(r)
    dir_y = wp.sin(r)
    
    t = wp.abs(direction - r)
    if t > wp.pi:
        t = wp.tau - t
    t = t / wp.pi
    t = wp.pow(t, 1.2)
    
    raw_amp = (2.0*t*t*t - 3.0*t*t + 1.0) * 1.0 + (-2.0*t*t*t + 3.0*t*t) * (1.0 - directionality)

    direction_data[d] = wp.vec3(dir_x, dir_y, raw_amp)
    wp.atomic_add(total_amp, 0, raw_amp)

@wp.kernel
def normalize_directions(
    direction_data: wp.array(dtype=wp.vec3),
    total_amp: wp.array(dtype=float),
    direction_count: int,
):
    d = wp.tid()
    entry = direction_data[d]
    average = total_amp[0] / float(direction_count)
    direction_data[d] = wp.vec3(entry[0], entry[1], entry[2] / average)

@wp.kernel
def flat_mesh_kernel(points: wp.array(dtype=wp.vec3), out_points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out_points[i] = points[i]


class Water:
    # USDRT
    stage = None
    water_path = "/World/Water"
    water_prim = None
    water_mesh = None

    # CPU mesh data
    orig_points = None

    # Warp arrays
    wp_points = None
    wp_out_points = None
    grid_x = None
    grid_y = None
    grid_indices = None
    grid_shape = None
    height_field_np = None
    direction_data = None

    # 1D ocean profile for spectrum-based waves
    profile = None
    directions = 128
    wave_scale = 1.0
    directional_spreading = 0.0
    wind_direction_deg = 0.0
    wind_speed_ms = 10.0
    fetch_km = 100.0
    gamma = 3.3
    water_depth_m = 50.0

    # Other wave vars
    t = 0.0
    is_waves_enabled = False
    is_initialized = False


    @staticmethod
    def update_waves_from_UI(is_waves_enabled, wave_scale,
                             directional_spreading, wind_direction_deg,
                             wind_speed_ms, fetch_km,
                             gamma, water_depth_m):
        Water.is_waves_enabled = is_waves_enabled
        Water.wave_scale = wave_scale
        Water.directional_spreading = directional_spreading
        Water.wind_direction_deg = wind_direction_deg
        Water.wind_speed_ms = wind_speed_ms
        Water.fetch_km = fetch_km
        Water.gamma = gamma
        Water.water_depth_m = water_depth_m


    @staticmethod
    def on_timeline_event(event):
        if event.type is int(omni.timeline.TimelineEventType.PLAY) and not Water.is_initialized:
            Water.water_prim = Usd.Stage.Attach(omni.usd.get_context().get_stage_id()).GetPrimAtPath(Sdf.Path(Water.water_path))
            Water.water_mesh = UsdGeom.Mesh(Water.water_prim)

            pts_attr = Water.water_mesh.GetPointsAttr()
            Water.orig_points_vt = pts_attr.Get()
            Water.orig_points = np.array(Water.orig_points_vt, dtype=np.float32)

            Water.wp_orig_points = wp.array(Water.orig_points, dtype=wp.vec3)
            Water.wp_points = wp.array(Water.orig_points, dtype=wp.vec3)
            Water.wp_out_points = wp.zeros(len(Water.orig_points), dtype=wp.vec3)
            Water.profile = wp.zeros(PROFILE_RES, dtype=wp.vec3)
            Water.direction_data = wp.zeros(Water.directions, dtype=wp.vec3)
            Water.direction_total = wp.zeros(1, dtype=wp.float32)

            grid_x = np.unique(Water.orig_points[:, 0]).astype(np.float32)
            grid_y = np.unique(Water.orig_points[:, 1]).astype(np.float32)
            if grid_x.size * grid_y.size == len(Water.orig_points):
                Water.grid_x = grid_x
                Water.grid_y = grid_y
                Water.grid_shape = (grid_y.size, grid_x.size)
                Water.grid_indices = np.lexsort((Water.orig_points[:, 0], Water.orig_points[:, 1]))
                Water.height_field_np = None
            else:
                Water.grid_x = None
                Water.grid_y = None
                Water.grid_shape = None
                Water.grid_indices = None
                Water.height_field_np = None

            Water.t = 0.0
            Water.is_initialized = True
        elif event.type is int(omni.timeline.TimelineEventType.STOP):
            wp.launch(kernel=flat_mesh_kernel,
                      dim=len(Water.orig_points),
                      inputs=[Water.wp_points, Water.wp_out_points]
            )
            verts = Vt.Vec3fArray(Water.wp_out_points.numpy().astype(np.float32))
            Water.water_mesh.GetPointsAttr().Set(verts)


    @staticmethod
    def on_physics_step(dt):
        Water.t += dt

        if Water.water_mesh is None:
            return

        if Water.is_waves_enabled:
            Water.update_mesh()
        else:
            wp.launch(kernel=flat_mesh_kernel,
                      dim=len(Water.orig_points),
                      inputs=[Water.wp_points, Water.wp_out_points]
            )
            verts = Vt.Vec3fArray(Water.wp_out_points.numpy().astype(np.float32))
            Water.water_mesh.GetPointsAttr().Set(verts)


    @staticmethod
    def update_mesh():
        if Water.wp_points is None or Water.wp_out_points is None or Water.profile is None:
            return

        wp.launch(
            kernel=update_profile,
            dim=(PROFILE_RES,),
            inputs=[
                Water.profile,
                PROFILE_RES,
                PROFILE_WAVENUM,
                MIN_WAVE_LENGTH,
                MAX_WAVE_LENGTH,
                PROFILE_EXTENT,
                Water.t,
                Water.wind_speed_ms,
                Water.fetch_km,
                Water.gamma,
                Water.water_depth_m
            ]
        )

        Water.direction_total.zero_()    #needs to reset each frame
        wp.launch(
            kernel=precompute_directions,
            dim= Water.directions,
            inputs=[
                Water.direction_data,
                Water.direction_total,
                Water.directional_spreading,
                Water.wind_direction_deg,
                Water.directions,
            ]
        )

        wp.launch(kernel=normalize_directions,
                  dim=Water.directions,
                  inputs=[Water.direction_data,
                  Water.direction_total,
                  Water.directions
            ]
        )

        wp.launch(
            kernel=update_points,
            dim=len(Water.orig_points),
            inputs=[
                Water.wp_points,
                Water.profile,
                PROFILE_RES,
                PROFILE_EXTENT,
                Water.wave_scale,
                wp.vec3(0.0, 0.0, 0.0),
                1.0,
                Water.direction_data,
                Water.directions,
                Water.wp_out_points,
            ]
        )

        pts = Water.wp_out_points.numpy().astype(np.float32)
        if Water.grid_indices is not None and Water.grid_shape is not None:
            Water.height_field_np = pts[Water.grid_indices, 2].reshape(Water.grid_shape)

        verts = Vt.Vec3fArray(pts)
        Water.water_mesh.GetPointsAttr().Set(verts)


    @staticmethod
    def get_wave_z(X, Y):
        x = np.asarray(X, dtype=np.float32)
        y = np.asarray(Y, dtype=np.float32)
        x, y = np.broadcast_arrays(x, y)

        if not Water.is_waves_enabled or Water.height_field_np is None or Water.grid_x is None or Water.grid_y is None:
            return np.zeros_like(x, dtype=np.float32)

        nx = Water.grid_shape[1]
        ny = Water.grid_shape[0]
        x_min = float(Water.grid_x[0])
        x_span = float(Water.grid_x[-1] - Water.grid_x[0])
        y_min = float(Water.grid_y[0])
        y_span = float(Water.grid_y[-1] - Water.grid_y[0])

        x_norm = np.mod((x - x_min) / x_span, 1.0) * float(nx)
        y_norm = np.mod((y - y_min) / y_span, 1.0) * float(ny)

        x0 = np.floor(x_norm).astype(np.int32) % nx
        x1 = (x0 + 1) % nx
        y0 = np.floor(y_norm).astype(np.int32) % ny
        y1 = (y0 + 1) % ny
        wx = x_norm - np.floor(x_norm)
        wy = y_norm - np.floor(y_norm)

        h00 = Water.height_field_np[y0, x0]
        h10 = Water.height_field_np[y0, x1]
        h01 = Water.height_field_np[y1, x0]
        h11 = Water.height_field_np[y1, x1]
        hx0 = (1.0 - wx) * h00 + wx * h10
        hx1 = (1.0 - wx) * h01 + wx * h11
        z = (1.0 - wy) * hx0 + wy * hx1

        if z.shape == ():
            return float(z)
        return z.astype(np.float32, copy=False)
