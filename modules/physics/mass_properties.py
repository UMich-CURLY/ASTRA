import numpy as np
from .volume import FULL_PROXY_HULL_VOLUME_M3, FULL_STBD_HULL_CENTROID, FULL_PORT_HULL_CENTROID

class MassProperties:
    def __init__(self,
                 L: float = 1.2,    #length
                 B: float = 0.91,   #beam
                 m_rb: float = 28.0,
                 m_hull: float = 7.5,   #mass of one hull in kg
                 V_hull: float = FULL_PROXY_HULL_VOLUME_M3,
                 com: np.ndarray = np.array([-0.309, 0.0, -0.047])):  

        self.L = L
        self.B = B
        self.m_rb = m_rb
        self.m_hull = m_hull
        self.V_hull = V_hull
        self.com = np.array(com, dtype=float)

    @staticmethod
    def parallel_axis(I_local, m, d, com):

        dx, dy, dz = d - com
        Ixx = I_local[0] + m * (dy**2 + dz**2)
        Iyy = I_local[1] + m * (dx**2 + dz**2)
        Izz = I_local[2] + m * (dx**2 + dy**2)

        return np.array([Ixx, Iyy, Izz])

    def inertia (self):
            
        hull_centroid_stbd = FULL_STBD_HULL_CENTROID  
        hull_centroid_port = FULL_PORT_HULL_CENTROID
        sensor_centroid = np.array([-0.389, 0.0, 0.078])   #solved for so it matches asv com assumption


        # some intermediate calcs
        # Inertia for the hulls
        r_hull = np.sqrt(self.V_hull / (np.pi * self.L))
        
        Ixx_hull = (self.m_hull * (r_hull**2) / 2)   #assume hull is cylindrical
        Iyy_hull = (self.m_hull * (3 * (r_hull ** 2) + (self.L ** 2))) / 12
        Izz_hull = Iyy_hull
        I_hull_local = np.array([Ixx_hull, Iyy_hull, Izz_hull])

        # Inertia for the sensor suite
        L_sensor = 1
        B_sensor = 0.35
        H_sensor = 0.25
        m_sensor = self.m_rb - (2 * self.m_hull)
        
        Ixx_sensor = m_sensor * ((B_sensor ** 2) + (H_sensor ** 2)) / 12
        Iyy_sensor = m_sensor * ((L_sensor ** 2) + (H_sensor ** 2)) / 12
        Izz_sensor = m_sensor * ((B_sensor ** 2) + (L_sensor ** 2)) / 12
        I_sensor_local = np.array([Ixx_sensor, Iyy_sensor, Izz_sensor])

        I_port = self.parallel_axis(I_hull_local, self.m_hull, hull_centroid_port, self.com)
        I_stbd = self.parallel_axis(I_hull_local, self.m_hull, hull_centroid_stbd, self.com)
        I_sensor = self.parallel_axis(I_sensor_local, m_sensor, sensor_centroid, self.com)

        total_inertia = I_port + I_stbd + I_sensor

        return total_inertia
    

