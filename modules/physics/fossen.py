import numpy as np

#nu = [u, v, w, p, q, r]
#tau = [X, Y, Z, K, M, N]

# M_RB·ν̇ + C_A(ν_r)·ν_r + D(ν_r)·ν_r + g(η) = τ

#M_RB·ν̇ and the implicit rigid-body Coriolis term — handled by PhysX itself, given the mass/CoM/inertia MassProperties (not computed here)
#C_A, D — this file.
#g(η) — asv.py::get_buoyancy_tau (nonlinear, wave-coupled, vertical-restoring only)
class ASVDynamics:
    def __init__(self,
                 #linear drag coefficients
                 X_u: float = 0.0,      # identified (was 5.4, hand-tuned)
                 Y_v: float = 35.0,
                 Z_w: float = 0.0,
                 K_p: float = 12.0,
                 M_q: float = 12.0,
                 N_r: float = 2.46,     # identified (was 20.0, hand-tuned)
                 
                 #quadratic drag coefficients
                 X_uu: float = 7.77,    # identified (was 5.0, hand-tuned)
                 Y_vv: float = 50.0,
                 Z_ww: float = 0.0,
                 K_pp: float = 4.0,
                 M_qq: float = 4.0,
                 N_rr: float = 9.01,    # identified (was 7.0, hand-tuneed)
                 M_A: np.ndarray = None):  

        #linear drag
        self.X_u = X_u
        self.Y_v = Y_v
        self.Z_w = Z_w
        self.K_p = K_p
        self.M_q = M_q
        self.N_r = N_r

        #quadratic drag
        self.X_uu = X_uu
        self.Y_vv = Y_vv
        self.Z_ww = Z_ww
        self.K_pp = K_pp
        self.M_qq = M_qq
        self.N_rr = N_rr

        self.eta = np.zeros(6)
        self.nu = np.zeros(6)

        self.M_A = M_A if M_A is not None else np.zeros((6, 6)) #added mass placeholder


    def damping_matrix(self, nu_r):

        u, v, w, p, q, r = nu_r
        
        D_lin = np.diag([self.X_u, self.Y_v, self.Z_w, self.K_p, self.M_q, self.N_r])
        D_nl = np.diag([self.X_uu * abs(u),
                        self.Y_vv * abs(v),
                        self.Z_ww * abs(w),
                        self.K_pp * abs(p),
                        self.M_qq * abs(q),
                        self.N_rr * abs(r),])

        return D_lin + D_nl

    def added_mass_coriolis_matrix(self, nu_r):
        return m2c(self.M_A, nu_r)

    def hydro_forces(self, nu, nu_water=None):

        nu_r = nu - (nu_water if nu_water is not None else np.zeros(6))

        D = self.damping_matrix(nu_r)
        C_A = self.added_mass_coriolis_matrix(nu_r)
        return -(D + C_A) @ nu_r

    
def m2c(M, nu):
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
    """

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    if (len(nu) == 6):      #  6-DOF model

        M11 = M[0:3,0:3]
        M12 = M[0:3,3:6] 
        M21 = M12.T
        M22 = M[3:6,3:6] 

        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 = np.matmul(M11,nu1) + np.matmul(M12,nu2)
        dt_dnu2 = np.matmul(M21,nu1) + np.matmul(M22,nu2)

        #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        C = np.zeros( (6,6) )    
        C[0:3,3:6] = -Smtrx(dt_dnu1)
        C[3:6,0:3] = -Smtrx(dt_dnu1)
        C[3:6,3:6] = -Smtrx(dt_dnu2)

    else:   # 3-DOF model (surge, sway and yaw)
        #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
        C = np.zeros( (3,3) ) 
        C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
        C[1,2] =  M[0,0] * nu[0] 
        C[2,0] = -C[0,2]       
        C[2,1] = -C[1,2]

    return C

#------------------------------------------------------------------------------

def Smtrx(a):
    """
    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b. 
    """
 
    S = np.array([ 
        [ 0, -a[2], a[1] ],
        [ a[2],   0,     -a[0] ],
        [-a[1],   a[0],   0 ]  ])

    return S

#------------------------------------------------------------------------------
