import numpy
import scipy.integrate as integrate
import tools # type: ignore
import method
import physics
import matplotlib.pyplot as plt

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
    return (0.5917*mesh.A_R*(T_b)**3) * numpy.ones(mesh.cell_centers.size)




numpy.set_printoptions(precision=8)

scale = tools.Scales()
mesh = tools.Discretization()

k_star = 27


mesh.groups = numpy.array([0.00000, 0.3, 0.6, 0.8, 1.2, 1.5, 1.8, 2.4, 
                           2.7, 3, 4, 5, 7, 9, 11, 15, 20, 1e4])*(1000/mesh.H)
mesh.dx = 0.4
mesh.t_stops = numpy.array([0, 2e-3, 2e-2]) * 1e-8
mesh.dt = 2e-3 * 1e-8 # seconds
mesh.eps = 1e-6
mesh.eps_c = 1e-3




T_prev  = (800/mesh.K)*numpy.ones((mesh.nx))
T_bound = (800/mesh.K)*numpy.ones((mesh.nx))
mesh.I_BC = numpy.zeros((mesh.ng, 2))
mesh.F_BC = numpy.zeros((mesh.ng, 2))
mesh.I_BC[:, 0] = 0.5*(physics.group_planck(mesh, T_bound))[:, 0]
mesh.F_BC[:, 0] = 0.25*(physics.group_planck(mesh, T_bound))[:, 0]

mesh.I_BC[:, 1] = 0.5*(physics.group_planck(mesh, T_bound))[:, 0]
mesh.F_BC[:, 1] = -0.25*(physics.group_planck(mesh, T_bound))[:, 0]

kappa   = group_FC_opacity(mesh, T_prev, k_star)

Cv    = FC_heatcap(1.0/mesh.K, mesh)
Q     = numpy.zeros((mesh.ng, mesh.nx))



sol_prev = tools.Transport_solution(mesh.nx, mesh.ng, numpy.zeros((mesh.ng, 4*mesh.nx)))
sol_prev.intensity[:,:] = tools.dbl(physics.group_planck(mesh, T_prev))


# Plot and compare to FC IMC results

T_out, I_out, unacc_iters = method.solve_diffusion(mesh, scale, group_FC_opacity, sol_prev, T_prev, Cv, accelerated=False)
T_out, I_out, acc_iters = method.solve_diffusion(mesh, scale, group_FC_opacity, sol_prev, T_prev, Cv, accelerated=True)

time_vales_ct = mesh.t_stops[1:]*mesh.C
time_labels = []

for t in time_vales_ct:
    time_labels.append(f"ct={t:.2f} cm")

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
lines = tools.LD_plottable(mesh, physics.ev_to_erg*I_out[-1].vec)
tools.plot_LD_groups(ax, mesh, lines.flux, range(0, mesh.ng))
plt.title(f"Grey flux, {time_labels[-1]}")
plt.xlabel("x [cm]")
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
plt.gca().set_ylim(bottom=0.0)
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
