import numpy as np
from pxr import UsdGeom, Gf

from .ocean_waves import Water

FULL_PROXY_HULL_VOLUME_M3 = 0.0565 / 2.0
REST_PROXY_HULL_VOLUME_M3 = 0.0285 / 2.0
TOTAL_REST_VOLUME_M3 = 2.0 * REST_PROXY_HULL_VOLUME_M3
REST_WATERLINE_Z_BODY = -0.11973

REST_STBD_HULL_CENTROID = np.array([-0.261645,  0.32979, -0.212761])
REST_PORT_HULL_CENTROID = np.array([-0.261645, -0.32979, -0.212761])

# Full (not submerged) hull volume centroids -- used for mass/inertia approximation
# (mass_properties.py), not buoyancy. Source: Blueboat ASV Info.xlsx, "Full volume" rows.
FULL_STBD_HULL_CENTROID = np.array([-0.239796,  0.33897, -0.155521])
FULL_PORT_HULL_CENTROID = np.array([-0.239796, -0.33897, -0.155521])

def proxy_sub_volume(local_pts, proxy_prim, body_prim):
    # Transform proxy points with the proxy prim itself, then express them in the body frame.
    xform_cache = UsdGeom.XformCache()
    proxy_to_world = xform_cache.GetLocalToWorldTransform(proxy_prim)
    body_to_world = xform_cache.GetLocalToWorldTransform(body_prim).RemoveScaleShear()
    world_to_body = body_to_world.GetInverse()

    world_pts = []
    body_pts = []
    for p in local_pts:
        world_pt = proxy_to_world.Transform(Gf.Vec3d(*p))
        body_pt = world_to_body.Transform(world_pt)
        world_pts.append([float(world_pt[0]), float(world_pt[1]), float(world_pt[2])])
        body_pts.append([float(body_pt[0]), float(body_pt[1]), float(body_pt[2])])

    world_pts = np.array(world_pts, dtype=np.float32)
    body_pts = np.array(body_pts, dtype=float)

    # Get water height at each point
    Z_water = Water.get_wave_z(world_pts[:, 0], world_pts[:, 1]) if Water.is_waves_enabled else 0.0

    # Get max draft allowed for each point in the body frame.
    top_hull = np.max(body_pts[:, 2])
    max_draft = top_hull - body_pts[:, 2]

    # Draft per point at the current wave elevation
    current_draft = np.clip(Z_water - world_pts[:, 2], 0, max_draft)

    # Rest draft per point clipped by max draft
    rest_draft = np.clip(REST_WATERLINE_Z_BODY - body_pts[:, 2], 0, max_draft)

    # Displacement to rest point, clipped within the hull points
    clipped_delta = np.clip(current_draft-rest_draft, -rest_draft, max_draft-rest_draft)

    # Convert summed draft displacement into displaced volume around rest
    summed_delta = np.sum(clipped_delta)
    vol_displacement = 0.0
    if summed_delta < 0:
        vol_displacement = (summed_delta / np.sum(rest_draft)) * REST_PROXY_HULL_VOLUME_M3
    elif summed_delta > 0:
        vol_displacement = (summed_delta / np.sum(max_draft - rest_draft)) * (FULL_PROXY_HULL_VOLUME_M3 - REST_PROXY_HULL_VOLUME_M3)

    # Rest volume + our displacement volume clipped to appropriate bounds
    vol = min(max((REST_PROXY_HULL_VOLUME_M3 + vol_displacement), 0.0), FULL_PROXY_HULL_VOLUME_M3)

    # Centroid of currently submerged volume in the body frame.
    centroid = np.array([0.0, 0.0, 0.0])
    if vol > 0:
        weighted_sum = np.sum(body_pts * current_draft[:, None], axis=0)
        weighted_sum[2] += 0.5 * np.sum(current_draft * current_draft)
        centroid = weighted_sum / np.sum(current_draft)
        # Centroid scaling: CAD_centroid_sub / approx_centroid_sub
        centroid *= (0.8678205589, 1.098222351, 0.93858922)

    return vol, centroid.tolist()


def floating_draft(prim, pts):
    pts = np.array(pts, dtype=float)
    
    # Transform body frame points to world frame
    xform_cache = UsdGeom.XformCache()
    body_to_world = xform_cache.GetLocalToWorldTransform(prim).RemoveScaleShear()
    
    world_pts = []
    for p in pts:
        world_pt = body_to_world.Transform(Gf.Vec3d(*p))
        world_pts.append([float(world_pt[0]), float(world_pt[1]), float(world_pt[2])])
    
    world_pts = np.array(world_pts, dtype=np.float32)
    
    # Get water height at each point
    Z_water = Water.get_wave_z(world_pts[:, 0], world_pts[:, 1]) if Water.is_waves_enabled else 0.0
    
    # Calculate draft at each point (water height - point height, clipped to 0)
    draft = np.clip(Z_water - world_pts[:, 2], 0, None)
    
    # Average draft across all points
    avg_draft = np.mean(draft) if len(draft) > 0 else 0.0
    
    return avg_draft