from typing import Callable, Optional

import numpy as np
import omni.usd
import omni.timeline
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
from usdrt import UsdGeom as UsdrtGeom
from usdrt import Usd as Usdrt
from usdrt import Sdf as Sdfrt

from .fossen import ASVDynamics
from .thrusters import ThrusterManager
from .keyboard_controller import KeyboardController
from .mass_properties import MassProperties

from .volume import proxy_sub_volume, TOTAL_REST_VOLUME_M3


class ASVController:
    def __init__(self,
                base_link_path: str = "/World/BlueBoat",
                proxy_hull_paths: list = ["/World/BlueBoat/stbd_hull/stbd_hull", "/World/BlueBoat/port_hull/port_hull"],
                port_prop: str = "/World/BlueBoat/port_prop",
                stbd_prop: str = "/World/BlueBoat/stbd_prop",
                mass: float = 28.0,
                thrust_step: float = 1.0,
                linear_damping: float = 0.0,
                angular_damping: float = 0.0,
                buoyancy_damping: float = 60.0,): 

        self.base_link_path = base_link_path
        self.proxy_hull_paths = proxy_hull_paths
        self.port_prop_path = port_prop
        self.stbd_prop_path = stbd_prop

        self.mass = mass
        self.mass_properties = MassProperties(m_rb=self.mass)
        self.com_offset_body = self.mass_properties.com

        self.dynamics = ASVDynamics()
        self.thrusters = ThrusterManager(com_offset=self.com_offset_body, asv_path=base_link_path)

        self.keyboard = KeyboardController(thrust_step=thrust_step)
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.buoyancy_damping = buoyancy_damping    #per-hull damping (=/= fossen heave damping since it acts at CoM of the ASV)
        self.buoyancy_scale = mass / (1000.0 * TOTAL_REST_VOLUME_M3)
        self.xform_cache = UsdGeom.XformCache()

        self.enabled = False
        self.force_api_applied = False
        self.logged_stage_units = False
        self.logged_submerged_hulls = set()

    # calculation to consider the CoM offset
    def to_com_frame(self, point_body):
        return np.array(point_body, dtype=float) - self.com_offset_body

    #cache clearing 
    def clear_cached_state(self):
        self.xform_cache.Clear()

    def clear_hull_wrench(self, stage=None):
        stage = stage or omni.usd.get_context().get_stage()
        if not stage:
            return

        force_api = PhysxSchema.PhysxForceAPI.Get(stage, self.base_link_path)
        if not force_api:
            return

        force_api.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        force_api.GetTorqueAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    ###turn on - shut off###
    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self.clear_cached_state()
            self.enabled = True
            self.force_api()
            self.clear_hull_wrench()

        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self.enabled = False
            self.force_api_applied = False
            self.clear_cached_state()
            self.thrusters.reset()
            self.keyboard.reset()
            self.clear_hull_wrench()

    def shutdown(self):
        self.clear_cached_state()
        self.clear_hull_wrench()
        self.keyboard.shutdown()

    def get_body_transforms(self, prim):
        self.clear_cached_state()
        body_to_world = self.xform_cache.GetLocalToWorldTransform(prim).RemoveScaleShear()
        world_to_body = body_to_world.GetInverse()
        return body_to_world, world_to_body

    ### get body frame velocity from physX ###

    def get_body_velocity(self, stage):
        prim = stage.GetPrimAtPath(self.base_link_path)
        if not prim:
            return np.zeros(6)

        rb = UsdPhysics.RigidBodyAPI(prim)
        lin_vel = rb.GetVelocityAttr().Get()          # world frame
        ang_vel = rb.GetAngularVelocityAttr().Get()   # world frame, deg/s

        if lin_vel is None or ang_vel is None:
            return np.zeros(6)

        #rotate the PhysX world-frame velocities into the hull frame directly.
        _, world_to_body = self.get_body_transforms(prim)
        body_lin_vel = world_to_body.TransformDir(Gf.Vec3d(float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])))
        body_ang_vel = world_to_body.TransformDir(Gf.Vec3d(float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])))

        return np.array([float(body_lin_vel[0]),
                        float(body_lin_vel[1]),
                        float(body_lin_vel[2]),
                        np.radians(float(body_ang_vel[0])),
                        np.radians(float(body_ang_vel[1])),
                        np.radians(float(body_ang_vel[2])),])

    ### apply PhysxForceAPI to existing prims ###
    def force_api(self):
        if self.force_api_applied:
            return

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        for path in [self.base_link_path, self.port_prop_path, self.stbd_prop_path]:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                print(f"[ASVController] WARNING: prim not found at {path}")
                continue

            force_api = PhysxSchema.PhysxForceAPI.Apply(prim)
            force_api.CreateModeAttr().Set("force")
            force_api.CreateForceEnabledAttr().Set(True)
            force_api.CreateWorldFrameEnabledAttr().Set(False)

        base_prim = stage.GetPrimAtPath(self.base_link_path)
        if base_prim:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(base_prim)
            physx_rb.CreateLockedPosAxisAttr().Set(0)
            physx_rb.CreateLockedRotAxisAttr().Set(0)
            physx_rb.CreateLinearDampingAttr().Set(self.linear_damping)
            physx_rb.CreateAngularDampingAttr().Set(self.angular_damping)

            mass_api = UsdPhysics.MassAPI.Apply(base_prim)
            mass_api.CreateMassAttr().Set(self.mass)
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(float(self.com_offset_body[0]),
                                                           float(self.com_offset_body[1]),
                                                           float(self.com_offset_body[2])))
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(float(self.mass_properties.inertia()[0]),
                                                              float(self.mass_properties.inertia()[1]),
                                                              float(self.mass_properties.inertia()[2])))
        
        self.force_api_applied = True

    def get_buoyancy_tau(self, stage, nu: np.ndarray):

    ## TO_DO: implement true relative velocity
    ## in fossen.py nu_water=0 anyways so not consequential yet 
        
        prim = stage.GetPrimAtPath(self.base_link_path)
        if not prim:
            return np.zeros(6)

        body_to_world, world_to_body = self.get_body_transforms(prim)
        linear_vel = nu[:3]
        angular_vel = nu[3:]
        tau_buoyancy = np.zeros(6)

        rho = 1000.0
        for proxy_path in self.proxy_hull_paths:
            proxy_prim = stage.GetPrimAtPath(proxy_path)
            proxy_prim_mesh = UsdGeom.Mesh(proxy_prim)
            proxy_prim_pts = proxy_prim_mesh.GetPointsAttr().Get()
            volume, centroid = proxy_sub_volume(proxy_prim_pts, proxy_prim, prim)

            if volume <= 0.0:
                continue

            centroid_body = np.array(centroid, dtype=float)
            centroid_rel_com = self.to_com_frame(centroid_body)
            centroid_body_vel = linear_vel + np.cross(angular_vel, centroid_rel_com)
            centroid_world_vel = body_to_world.TransformDir(Gf.Vec3d(*centroid_body_vel))   

            buoyancy_force = rho * 9.81 * volume * self.buoyancy_scale
            damping_force = -self.buoyancy_damping * float(centroid_world_vel[2])
            force_mag = buoyancy_force + damping_force
            if force_mag <= 0.0:
                continue

            body_force = world_to_body.TransformDir(Gf.Vec3d(0.0, 0.0, float(force_mag)))
            body_force_np = np.array([float(body_force[0]), float(body_force[1]), float(body_force[2])])
            torque_contribution = np.cross(centroid_rel_com, body_force_np)
            tau_buoyancy[:3] += body_force_np
            tau_buoyancy[3:] += torque_contribution

        return tau_buoyancy

    ### apply body-frame wrench to the hull ###
    def apply_hull_forces(self, stage, tau_total: np.ndarray):
        hydro_prim = stage.GetPrimAtPath(self.base_link_path)
        if not hydro_prim:
            return

        force_api = PhysxSchema.PhysxForceAPI.Get(stage, self.base_link_path)
        if not force_api:
            return
        
        force_attr = force_api.GetForceAttr()
        torque_attr = force_api.GetTorqueAttr()
        force_attr.Set(Gf.Vec3f(float(tau_total[0]), float(tau_total[1]), float(tau_total[2])))
        torque_attr.Set(Gf.Vec3f(float(tau_total[3]), float(tau_total[4]), float(tau_total[5])))

    ### physics step magic ###
    def on_physics_step(self, dt: float):
        if not self.enabled or not self.force_api_applied:
            return

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        
        #read velocity
        nu = self.get_body_velocity(stage)

        #get thruster commands
        port_cmd, stbd_cmd = self.keyboard.get_thruster_commands()
        self.thrusters.set_commands(port_cmd, stbd_cmd)

        #calculate fossen forces
        tau_hydro = self.dynamics.hydro_forces(nu)

        #thruster contribution in body frame
        tau_thrusters = self.thrusters.get_total_tau(dt)

        #buoyancy / restoring forces from simple twin-hull support points
        tau_buoyancy = self.get_buoyancy_tau(stage, nu)

        #apply total forces on the hull
        self.apply_hull_forces(stage, tau_hydro + tau_thrusters + tau_buoyancy)

        self.get_telemetry()

    ### UI read outs ###
    def get_telemetry(self) -> dict:
        stage = omni.usd.get_context().get_stage()
        nu = self.get_body_velocity(stage) if stage else np.zeros(6)
        port_cmd, stbd_cmd = self.keyboard.get_thruster_commands()

        return {"surge_vel": nu[0],
                "sway_vel": nu[1],
                "heave_vel": nu[2],
                "roll_rate_deg": np.degrees(nu[3]),
                "pitch_rate_deg": np.degrees(nu[4]),
                "yaw_rate_deg": np.degrees(nu[5]),
                "port_cmd": port_cmd,
                "stbd_cmd": stbd_cmd,
                "port_force_N": self.thrusters.port.get_force(),
                "stbd_force_N": self.thrusters.stbd.get_force(),}
