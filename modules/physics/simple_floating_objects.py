import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf

from .volume import floating_draft
from .ocean_waves import Water

class FoatingObjects:
    def __init__(self,
                folder_paths: list = ["/World/Branches", "/World/TrashBottles", "/World/Buoys"],
                buoyancy_scale: float = 1.0,
                buoyancy_damping: float = 10.0,
                sample_points: list = [[0.0, 0.0, 0.0]],
                update_hz: float = 10.0):

        self.folder_paths = folder_paths
        self.buoyancy_scale = buoyancy_scale
        self.buoyancy_damping = buoyancy_damping
        self.sample_points = sample_points
        self.update_hz = update_hz
        
        self.disabled = False

        self._accumulated_time = 0.0
        self._xform_cache = UsdGeom.XformCache()


    ### get transforms between body and world frames ###
    def get_body_transforms(self, prim):
        body_to_world = self._xform_cache.GetLocalToWorldTransform(prim).RemoveScaleShear()
        world_to_body = body_to_world.GetInverse()
        return body_to_world, world_to_body
    

    ### Reset each prim's PhysxForceAPI values ###
    def reset_APIs(self):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        # Go through all folder paths containing simple floating prims
        for path in self.folder_paths:
            parent = stage.GetPrimAtPath(path)
            if not parent:
                print(f"[ASVController] WARNING: folder not found at {path}")
            
            # Reset the prim's values. We could apply PhysxRigidBodyAPI and PhysxForceAPI
            # here, but that also means setting the correct lin damping, ang damping, and mass...
            prims = parent.GetChildren()
            for prim in prims:
                force_api = PhysxSchema.PhysxForceAPI.Get(stage, prim.GetPath())
                if not force_api:
                    print(f"[ASVController] WARNING: {prim} has no PhysxForceAPI")

                force_api.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                force_api.GetTorqueAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


    ### turn on - shut off ###
    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._xform_cache.Clear()
            self._accumulated_time = 0.0
            self.disabled = False
            self.reset_APIs()

        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._xform_cache.Clear()
            self._accumulated_time = 0.0
            self.disabled = False
            self.reset_APIs()


    ### physics step ###
    def on_physics_step(self, dt: float):
        # Return early if on_physics_step is not necessary
        if not Water.is_waves_enabled and self.disabled:
            return

        # Only update simple floating prims based on 'update_hz'
        # If Hz is too small, it can cause problems!
        if (1.0/self.update_hz) > 0.0:
            self._accumulated_time += dt
            if self._accumulated_time < (1.0/self.update_hz):
                return
            self._accumulated_time -= (1.0/self.update_hz)

        stage = omni.usd.get_context().get_stage()

        # Go through all folder paths containing simple floating prims
        for path in self.folder_paths:
            parent = stage.GetPrimAtPath(path)
            if not parent:
                print(f"[ASVController] WARNING: folder not found at {path}")

            # Assign different sample points and buoyancy scales for each group of prims
            if parent.GetName() == 'Branches':
                self.buoyancy_scale = 1
                self.sample_pts = [[ 0.0, 0.0, 0.0],
                                   [ 0.5, 0.0, 0.0],
                                   [ 1.0, 0.0, 0.0],
                                   [-0.5, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0]]
                
            if parent.GetName() == 'TrashBottles':
                self.buoyancy_scale = 0.005
                self.sample_pts = [[ 0.0,  0.0, -0.15],
                                   [ 0.1,  0.0, -0.15],
                                   [-0.1,  0.0, -0.15],
                                   [ 0.0,  0.1, -0.15],
                                   [ 0.0, -0.1, -0.15]]
                
            if parent.GetName() == 'Buoys':
                self.buoyancy_scale = 0.05
                self.sample_pts = [[ 0.0,  0.0, -0.30],
                                   [ 0.1,  0.0, -0.25],
                                   [-0.1,  0.0, -0.25],
                                   [ 0.0,  0.1, -0.25],
                                   [ 0.0, -0.1, -0.25]]

            # Iterate through all prims in the folder and apply a simple buoyancy force
            prims = parent.GetChildren()
            for prim in prims:
                force_api = PhysxSchema.PhysxForceAPI.Get(stage, prim.GetPath())
                if not force_api:
                    print(f"[ASVController] WARNING: {prim} has no PhysxForceAPI")
                    continue

                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                if not Water.is_waves_enabled:
                    rigid_body_api.CreateKinematicEnabledAttr().Set(True)
                    self.disabled = True
                    continue  
                elif rigid_body_api.GetKinematicEnabledAttr().Get():
                    rigid_body_api.CreateKinematicEnabledAttr().Set(False)
                    self.disabled = False

                #check floating draft and reset force to (0,0,0) if not underwater
                avg_draft = floating_draft(prim, self.sample_pts)
                if avg_draft <= 0.0:
                    force_api.GetForceAttr().Set(Gf.Vec3d(0.0, 0.0, 0.0))

                #buoyancy force based on avg draft and a coef
                rho = 1000.0
                buoyancy_force = rho * 9.81 * avg_draft * self.buoyancy_scale

                body_to_world, world_to_body = self.get_body_transforms(prim)
                #damping force (objects with a small mass)
                prim_body_vel = UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr().Get() # + angular_vel

                if prim_body_vel is not None:
                    prim_world_vel = body_to_world.TransformDir(Gf.Vec3d(*prim_body_vel))
                    damping_force = -self.buoyancy_damping * float(prim_world_vel[2])

                    #force magnitude from buoyancy and damping
                    force_mag = buoyancy_force + damping_force
                    if force_mag <= 0.0:
                        force_api.GetForceAttr().Set(Gf.Vec3d(0.0, 0.0, 0.0))

                    #apply the force in the body frame
                    body_force = world_to_body.TransformDir(Gf.Vec3d(0.0, 0.0, float(force_mag)))
                    force_api.GetForceAttr().Set(body_force)

