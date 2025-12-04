import numpy
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import physics
from itertools import cycle

from physics import C, SIG_R, A_R, H


b_0i = numpy.array([[-0.25, 0.25, 0,     0   ],
                    [0,     0,    0.25, -0.25]])
    
b_0f = numpy.array([[-0.5,  -0.5, 0,   0  ],
                    [0,     0,    0.5, 0.5]])

b_1i = numpy.array([[-0.5, -0.5,    0,   0  ],
                    [ 0,    0,      0.5, 0.5]])

b_1f = numpy.array([[-0.75, 0.75, 0,     0   ],
                    [ 0,    0,    0.75, -0.75]])

a_f  = numpy.array([[0,  0.5,  0.5, 0],
                     [0, -0.5, -0.5, 0]])

M    = (1/6)*numpy.array([[2, 1],
                              [1, 2]])
M_wide = (1/6)*numpy.array([[0, 2, 1, 0],
                              [0, 1, 2, 0]])

colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

def dbl(array):
    if array.ndim > 1:
        return numpy.repeat(array, 2, axis=1) # repeats along second axis, spatial
    else:
        return numpy.repeat(array, 2)
    # doubles a constant, to reshape for L and R

class Transport_solution:
    def __init__(self, mesh):
        self.vec = numpy.zeros((mesh.ng, 4*mesh.nx))
        self._nx =  numpy.copy(mesh.nx)
        self._ng = numpy.copy(mesh.ng)

    
    @property
    def intensity(self):
        return self.vec[:, :2*self._nx]
    
    @intensity.setter
    def intensity(self, value):
        self.vec[:, :2*self._nx] = value

    @property
    def flux(self):
        return self.vec[:, 2*self._nx:]
    
    @flux.setter
    def flux(self, value):
        self.vec[:, 2*self._nx:] = value

    @property
    def cell_center_i(self):
        return 0.5 * (self.intensity[:, 1::2] + self.intensity[:, 0::2])
    
        

class Scales:
    t = 1
    I = 1
    B = 1
    E = 1

class Discretization:
    def __init__(self):
        self.groups = numpy.array([0.000, 1e10])*(1000/H)
        self.x_length = 4
        self.t_stops = numpy.array([0, 1e-2, 2e-2])
        self.dx = 0.4 # cm
        self.dt = 2e-3 # shakes
        self.I_BC = numpy.zeros((1, 2))
        self.F_BC = numpy.zeros((1, 2))
        self.eps = 1e-6

        self.H = numpy.float128(4.135667696e-15) # eV * s
        self.K = numpy.float128(8.617333262e-5)  # eV / K
        self.C = numpy.float128(2.99792458e10)   # cm / s
        self.SIG_R = numpy.float128(3.53916934e7) # eV / (cm^2 * s * K^4)
        self.A_R = numpy.float128(4.72215928e-3)  # eV / (cm^2 * s * K^4)
        # self.nx = 10
        # self.nt = numpy.array([50, 100])

        # self.cell_centers = numpy.linspace(0.2, 3.8, 10)
        # self.cell_edges   = numpy.linspace(0, 4, 11)

        # TODO -- change all these to manual definition as in below. Right now recursive definition.

    def rescale(self, rho : Scales):
        self.C = C * rho.t
        self.SIG_R = SIG_R / rho.B
        self.A_R = A_R / rho.B

    @property 
    def ng(self):
        return len(self.groups) - 1
    @property
    def nx(self):
        return int(self.x_length / self.dx)
    
    @property
    def nt(self):
        return numpy.round(numpy.diff(self.t_stops)/self.dt, decimals=0).astype(int)

    @property
    def cell_centers(self):
        return numpy.linspace((self.dx/2), self.x_length - (self.dx/2), self.nx)
        
    
    @property
    def cell_edges(self):
        return numpy.linspace(0, self.x_length, self.nx+1)

# class Local_system():
#     def __init__(self):
#         self.range = [0, 0] # domain where this stencil should be applied.
#                             # from (i = 0 + range[0]) to (i = nx - range[1])
#         self.size  = [0, 0] # how many elements are [before, after] the regular
#                             # [xL, xR]?
#         self.mat   = numpy.array([[[0]], [[0]]]) # stencil values
#                                 # [action on F]
#                                        # [Action on I]

#         # maybe remove these
#         self.row   = 0      # which equations 
#         self.col   = 0      # which dimension does this stencil apply to

class Global_system():
    def __init__(self):
        self.mat = sparse.csr_array(numpy.array([[]]))
        self.src = sparse.csr_array(numpy.array([]))


class MG_coefficients:
    def __init__(self, mesh : Discretization):
        self.sig_a = 1
        self.eta = numpy.zeros((mesh.nx))
        self.sig_f = numpy.zeros((mesh.ng, mesh.nx))
        self.chi = numpy.zeros((mesh.ng, mesh.nx))
        self.q   = numpy.zeros((mesh.ng, mesh.nx))
        self.kappa = numpy.zeros((mesh.ng, mesh.nx))
        self.beta  = numpy.zeros((mesh.ng, mesh.nx))
        self.db_dt = numpy.zeros((mesh.ng, mesh.nx))
        self.S = numpy.zeros((mesh.ng, 2*mesh.nx))
        self.D = numpy.zeros((mesh.ng, mesh.nx))


    def assign(self, mesh : Discretization, kappa, sol_prev : Transport_solution, T_prev, Cv, Q):
        self.kappa[:,:] = kappa.copy()
        I_prev = sol_prev.intensity.copy()
        self.db_dt[:,:] = physics.group_dB_dT(mesh, T_prev)
        self.beta[:,:] = physics.group_planck(mesh, T_prev)

        self.sig_a = 1/(mesh.C*mesh.dt)
        self.sig_f[:,:] = kappa.copy()

        self.chi[:,:] = (kappa * self.db_dt) / numpy.tile(numpy.sum(kappa * self.db_dt, axis=0), reps = (mesh.ng, 1))
        self.eta[:] = (numpy.sum(kappa * self.db_dt, axis=0)
                    /
                    ((Cv / mesh.dt)+
                        numpy.sum(kappa * self.db_dt, axis=0))
                )
        self.q[:,:]   = ((kappa * self.beta) + 
               self.eta*self.chi*(Q - numpy.sum(kappa * self.beta, axis=0)))
        self.S[:,:] = dbl(self.q) + I_prev/(mesh.C*mesh.dt)
        self.D[:,:] = 1/(3*kappa)


class LD_plottable:

    def __init__(self, mesh, solution):
        self.intensity = []
        self.flux = []

        grey_I_L = numpy.zeros(mesh.nx)
        grey_I_R = numpy.zeros(mesh.nx)
        grey_F_L = numpy.zeros(mesh.nx)
        grey_F_R = numpy.zeros(mesh.nx)

        for g in range(0, mesh.ng):
            I_L = solution[g, 0:2*mesh.nx:2]
            I_R = solution[g, 1:2*mesh.nx:2]
            F_L = solution[g, 2*mesh.nx:4*mesh.nx:2]
            F_R = solution[g, 2*mesh.nx + 1:4*mesh.nx:2]

            grey_I_L += I_L
            grey_I_R += I_R
            grey_F_L += F_L
            grey_F_R += F_R

            segments_I = numpy.array([
                [[mesh.cell_edges[i], I_L[i]], [mesh.cell_edges[i+1], I_R[i]]]
                for i in range(mesh.nx)
            ])

            segments_F = numpy.array([
                [[mesh.cell_edges[i], F_L[i]], [mesh.cell_edges[i+1], F_R[i]]]
                for i in range(mesh.nx)
            ])

            self.intensity.append(LineCollection(segments_I))
            self.flux.append(LineCollection(segments_F))

        grey_segments_I = numpy.array([
            [[mesh.cell_edges[i], grey_I_L[i]], [mesh.cell_edges[i+1], grey_I_R[i]]]
            for i in range(mesh.nx)
        ])

        grey_segments_F = numpy.array([
            [[mesh.cell_edges[i], grey_F_L[i]], [mesh.cell_edges[i+1], grey_F_R[i]]]
            for i in range(mesh.nx)
        ])

        self.grey_intensity = LineCollection(grey_segments_I)
        self.grey_flux = LineCollection(grey_segments_F)

def plot_LD_groups(mesh : Discretization, lines: LineCollection, groups=[0]):
    legend_proxies = []
    legend_labels = []

    for g in groups:
        c = lines[g]
        col = next(colors)
        c.set_color(col)
        plt.gca().add_collection(c)

        legend_proxies.append(Line2D([0],[0], color=col, linestyle = "-"))
        legend_labels.append(f"g={g}")
    plt.legend(legend_proxies, legend_labels)
    return 0
    