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


mesh.t_stops = numpy.array([0, 2e-3, 2e-2, 5e-2, 1e-1, 2e-2, 3e-1]) * 1e-8
mesh.dt = 2e-3 * 1e-8 # seconds


mesh.eps = 1e-4



# testing-- this code should be deleted




T_prev  = (1/mesh.K)*numpy.ones((mesh.nx))
T_bound = (1000/mesh.K)*numpy.ones((mesh.nx))
kappa   = group_FC_opacity(mesh, T_prev, k_star)


Cv    = FC_heatcap(1.0/mesh.K, mesh)



Q     = numpy.zeros((mesh.ng, mesh.nx))

mesh.I_BC[:, 0] = (physics.group_planck(mesh, T_bound))[:, 0]
mesh.F_BC[:, 0] = 0.5*(physics.group_planck(mesh, T_bound))[:, 0]

# mesh.I_BC[:, 1] = (physics.group_planck(mesh, T_bound))[:, 0]
# mesh.F_BC[:, 1] = -0.5*(physics.group_planck(mesh, T_bound))[:, 0]

sol_prev = tools.Transport_solution(mesh.nx, mesh.ng, numpy.zeros((mesh.ng, 4*mesh.nx)))
sol_prev.intensity[:,:] = tools.dbl(physics.group_planck(mesh, T_prev))




T_out, I_out, unacc_iters = method.solve_diffusion(mesh, scale, group_FC_opacity, sol_prev, T_prev, Cv, accelerated=False)
T_out, I_out, acc_iters = method.solve_diffusion(mesh, scale, group_FC_opacity, sol_prev, T_prev, Cv, accelerated=True)

time_vales_ct = mesh.t_stops[1:]*mesh.C
time_labels = []

for t in time_vales_ct:
    time_labels.append(f"ct={t:.2f} cm")

FC_x = [0.2,0.6,1,1.4,1.8,2.2,2.6,3,3.4,3.8]
FC_T = 1000*numpy.array([[0.795,0.64,0.425,0.23,0.15,0.09,0.06,0.04,0.03,0.02],
        [0.87,0.85,0.78,0.685,0.585,0.495,0.35,0.27,0.21,0.16],
        [0.93,0.945,0.92,0.89,0.89,0.87,0.82,0.78,0.715,0.605]])

plt.figure()
ax = plt.gca()
for i in range(0, len(I_out)):
    lines = tools.LD_plottable(mesh, physics.ev_to_erg*I_out[i].vec)
    tools.plot_LD_grey(ax, lines.grey_intensity)
plt.title(f"Grey intensity over time")
plt.xlabel("x [cm]")
plt.legend(time_labels)
plt.autoscale()

plt.figure()
ax = plt.gca()
for i in range(0, len(I_out)):
    lines = tools.LD_plottable(mesh, (1/mesh.C)*physics.ev_to_erg*I_out[i].vec)
    tools.plot_LD_grey(ax, lines.grey_intensity)
plt.title(f"Energy density over time")
plt.xlabel("x [cm]")
plt.legend(time_labels)
plt.autoscale()



plt.figure()
for i in range(0, len(T_out)):
    plt.plot(mesh.cell_centers, mesh.K*T_out[i], label = f"t={mesh.t_stops[i+1]:.1e} s")
for t in range(0, FC_T.shape[0]):
    plt.scatter(FC_x, FC_T[t])
plt.legend()
plt.xlabel("x [cm]")
plt.ylabel("T [eV]")
plt.title("Temperature over time")


s = numpy.sum(mesh.nt)
plt.figure()
plt.plot(mesh.C*numpy.linspace(1, s, s)*mesh.dt, unacc_iters[1], label="Unaccelerated")
plt.plot(mesh.C*numpy.linspace(1, s, s)*mesh.dt, acc_iters[1], label="Accelerated")
plt.xlabel("ct [cm]")
plt.ylabel("count")
plt.title("Inner Iterations")
plt.legend()

plt.figure()
plt.plot(mesh.C*numpy.linspace(1, s, s)*mesh.dt, unacc_iters[0], label="Unaccelerated")
plt.plot(mesh.C*numpy.linspace(1, s, s)*mesh.dt, acc_iters[0], label="Accelerated")
plt.xlabel("ct [cm]")
plt.ylabel("count")
plt.title("Outer Iterations")
plt.legend()

plt.figure()
plt.plot(mesh.C*numpy.linspace(1, s, s)*mesh.dt, unacc_iters[0], label="Outer Iterations")
plt.plot(mesh.C*numpy.linspace(1, s, s)*mesh.dt, unacc_iters[1], label="Inner Iterations")
plt.xlabel("ct [cm]")
plt.ylabel("count")
plt.title("Uanccelerated Solve")
plt.legend()
lim = plt.gca().get_ylim()

plt.figure()
plt.plot(mesh.C*numpy.linspace(1, s, s)*mesh.dt, acc_iters[0], label="Outer Iterations")
plt.plot(mesh.C*numpy.linspace(1, s, s)*mesh.dt, acc_iters[1], label="Inner Iterations")
plt.xlabel("ct [cm]")
plt.ylabel("count")
plt.title("Accelerated Solve")
plt.legend()
plt.ylim(lim)

plt.show()
