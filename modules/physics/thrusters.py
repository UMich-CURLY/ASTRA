import numpy as np
import omni.usd
from pxr import UsdGeom, Gf

from .algae import Algae


class Thruster:
    def __init__(self,
                position: np.ndarray,
                direction: np.ndarray = None,
                com_offset: np.ndarray = None,
                max_forward: float = 20.0,
                max_reverse: float = -10.0):

        self.position = np.array(position, dtype=float)
        if self.position.size == 2:
            self.position = np.append(self.position, 0.0)

        self.direction = np.array(direction if direction is not None else [1.0, 0.0, 0.0], dtype=float)
        direction_norm = np.linalg.norm(self.direction)
        if direction_norm == 0.0:
            self.direction = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self.direction = self.direction / direction_norm

        self.com_offset = np.array(com_offset if com_offset is not None else [0.0, 0.0, 0.0], dtype=float)

        self.max_forward = max_forward
        self.max_reverse = max_reverse
        self.command = 0.0

        self.algae_penalty = 1.0

    def get_force(self):
        cmd = np.clip(self.command, -1.0, 1.0)
        if cmd >= 0:
            return self.max_forward * cmd
        else:
            return -self.max_reverse * cmd

    def get_force_vector(self):
        return self.get_force() * self.direction

    def get_force_and_moment(self, force):
        moment_arm = self.position - self.com_offset
        moment = np.cross(moment_arm, force)
        return np.concatenate((force, moment))

class ThrusterManager:
    def __init__(self,
                port_position: list = [-0.84307,  0.298, -0.33795],
                stbd_position: list = [-0.84307, -0.298, -0.33795],
                port_direction: list = [1.0, 0.0, 0.0],
                stbd_direction: list = [1.0, 0.0, 0.0],
                com_offset: list = [0.0, 0.0, 0.0],
                max_forward: float = 20.0,
                max_reverse: float = -10.0,
                asv_path: str = "/World/BlueBoat"):
                
        self.port = Thruster(position=port_position,
                            direction=port_direction,
                            com_offset=com_offset,
                            max_forward=max_forward,
                            max_reverse=max_reverse,)
        
        self.stbd = Thruster(position=stbd_position,
                            direction=stbd_direction,
                            com_offset=com_offset,
                            max_forward=max_forward,
                            max_reverse=max_reverse,)
        
        self.asv_path = asv_path

    def set_commands(self, port_cmd, stbd_cmd):
        self.port.command = port_cmd
        self.stbd.command = stbd_cmd

    def get_total_tau(self, dt):
        port_force_moment = self.port.get_force_and_moment(self.port.get_force_vector() * self.algae_penalty(self.port, dt))
        stbd_force_moment = self.stbd.get_force_and_moment(self.stbd.get_force_vector() * self.algae_penalty(self.stbd, dt))
        return port_force_moment + stbd_force_moment

    def algae_penalty(self, thruster, dt):  #thrust behavior in 'algae'
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.asv_path)
        body_to_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim).RemoveScaleShear()
        world_pos = body_to_world.Transform(Gf.Vec3d(*thruster.position.tolist()))

        if Algae.is_in_algae(world_pos):
            thruster.algae_penalty += 200*dt
            return (1.0/thruster.algae_penalty)
        thruster.algae_penalty = 1.0
        return 1.0

    def reset(self):
        self.port.command = 0.0
        self.stbd.command = 0.0
