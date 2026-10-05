import numpy as np
import omni.usd
from pxr import UsdGeom


class Algae:
    algae_folder_path = "/World/Algae"
    xform_cache = UsdGeom.XformCache()


    @staticmethod
    def get_body_transforms(self, prim):
        body_to_world = self.xform_cache.GetLocalToWorldTransform(prim).RemoveScaleShear()
        world_to_body = body_to_world.GetInverse()
        return body_to_world, world_to_body
    

    @staticmethod
    def is_in_algae(position):
        stage = omni.usd.get_context().get_stage()
        parent = stage.GetPrimAtPath(Algae.algae_folder_path)
        prims = parent.GetChildren()

        for prim in prims:
            translate = prim.GetAttribute("xformOp:translate").Get()

            diameter = (UsdGeom.Mesh(prim).GetExtentAttr().Get()[1][0])

            algae_x = float(translate[0])
            algae_y = float(translate[1])

            dist = np.sqrt((position[0] - algae_x)**2 + (position[1] - algae_y)**2)

            if dist < (diameter):
                return True
        
        return False

