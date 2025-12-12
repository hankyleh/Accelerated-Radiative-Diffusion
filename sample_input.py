import numpy
import scipy.integrate as integrate
import scipy.sparse as sparse
import tools # type: ignore
from physics import A_R, C# type: ignore
import method # type: ignore
import physics # type: ignore
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
import copy



numpy.set_printoptions(precision=8)


# Define temperature-dependent coefficients as functions



def FC_opacity(T, nu, k0, mesh):
    def num_func(n):
        return (n**7)*(1-numpy.exp(-mesh.H*n/(mesh.K*T)))**-3
    def dem_func(n):
        return (n**4)*(1-numpy.exp(-mesh.H*n/(mesh.K*T)))**-2
    
    

    numerator, err = integrate.quad(num_func, nu[0], nu[1])
    denominator, err = integrate.quad(dem_func, nu[0], nu[1])

    if numerator == 0:
        print(T)
        print(nu)
        print(numerator)
        print(denominator)

    return (1000**3)*k0 * denominator/(numerator*((mesh.H)**3))

def group_FC_opacity(mesh, T, k0):
    kappa = numpy.zeros((mesh.ng, mesh.nx))

    for g in range(0, mesh.ng):
        for x in range(0, mesh.nx):
            kappa[g, x] = FC_opacity(T[x], mesh.groups[g:g+2], k0, mesh)
    return kappa

def FC_heatcap(T_b, mesh):
    return (0.5917*A_R*(T_b)**3) * numpy.ones(mesh.cell_centers.size)





scale = tools.Scales()
mesh = tools.Discretization()

k_star = 27

# scale.I = 1

mesh.groups = numpy.array([0.00000, 0.3, 0.6, 0.8, 1.2, 1.5, 1.8, 2.4, 
                           2.7, 3, 4, 5, 7, 9, 11, 15, 20, 1e4])*(1000/mesh.H)

# mesh.groups = numpy.array([0.000001,  1e4])*(1000/mesh.H)
mesh.dx = 0.4



mesh.I_BC = numpy.zeros((mesh.ng, 2))
mesh.F_BC = numpy.zeros((mesh.ng, 2))


mesh.t_stops = numpy.array([0, 2e-3, 1e-2]) * 1e-8
mesh.dt = 2e-3 * 1e-8 # seconds


mesh.eps = 1e-4




# testing-- this code should be deleted




T_prev  = (500/mesh.K)*numpy.ones((mesh.nx))
T_bound = (1000/mesh.K)*numpy.ones((mesh.nx))
kappa   = group_FC_opacity(mesh, T_prev, k_star)


Cv    = FC_heatcap(1.0/mesh.K, mesh)



Q     = numpy.zeros((mesh.ng, mesh.nx))

mesh.I_BC[:, 0] = (physics.group_planck(mesh, T_bound))[:, 0]
mesh.F_BC[:, 0] = 0.5*(physics.group_planck(mesh, T_bound))[:, 0]

sol_prev = tools.Transport_solution(mesh.nx, mesh.ng, numpy.zeros((mesh.ng, 4*mesh.nx)))
sol_prev.intensity[:,:] = tools.dbl(physics.group_planck(mesh, T_prev))




T_out, I_out = method.solve_unaccelerated(mesh, scale, group_FC_opacity, sol_prev, T_prev, Cv)



plt.figure()
ax = plt.gca()
for i in range(0, len(I_out)):
    lines = tools.LD_plottable(mesh, I_out[i].vec)
    tools.plot_LD_groups(ax, mesh, lines.intensity, [0])
plt.title(f"Intensity over time")
plt.autoscale()

plt.figure()
ax = plt.gca()

lines = tools.LD_plottable(mesh, I_out[-1].vec)
tools.plot_LD_groups(ax, mesh, lines.intensity, range(0, mesh.ng))
plt.title(f"Final groups intensity")
plt.autoscale()

# plt.figure()
# ax = plt.gca()
# lines = tools.LD_plottable(mesh, I_out[-1].vec)
# tools.plot_LD_groups(ax, mesh, lines.flux, range(0, mesh.ng))
# plt.autoscale()
# # plt.close()

plt.figure()
for i in range(0, len(T_out)):
    plt.plot(mesh.cell_centers, mesh.K*T_out[i], label = f"t={mesh.t_stops[i+1]:.1e} s")
plt.legend()
plt.xlabel("X [cm]")
plt.ylabel("T [eV]")
plt.title("Temperature over time")

plt.show()
