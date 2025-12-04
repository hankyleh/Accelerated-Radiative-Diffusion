import numpy
import scipy.integrate as integrate
import scipy.sparse as sparse
import tools # type: ignore
from physics import H, K, A_R, C# type: ignore
import method # type: ignore
import physics # type: ignore
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle




# Define temperature-dependent coefficients as functions



def FC_opacity(T, nu, k0):
    def num_func(n):
        return (H**3)*(n**7)*(1-numpy.exp(-H*n/(K*T)))**-2
    def dem_func(n):
        return (n**4)*(1-numpy.exp(-H*n/(K*T)))**-2

    numerator, err = integrate.quad(num_func, nu[0], nu[1])
    denominator, err = integrate.quad(dem_func, nu[0], nu[1])
    return k0 * denominator/numerator

def group_FC_opacity(mesh, T, k0):
    kappa = numpy.zeros((mesh.ng, mesh.nx))

    for g in range(0, mesh.ng):
        for x in range(0, mesh.nx):
            kappa[g, x] = FC_opacity(T[x], mesh.groups[g:g+2], k0)
    return kappa

def FC_heatcap(T_b, mesh):
    return (0.5917*A_R*(T_b)**3) * numpy.ones(mesh.cell_centers.size)


# Units for scaling computations, such that
# x_physical = x_working * scale_x

scale = tools.Scales()
mesh = tools.Discretization()

k_star = 27

# scale.I = 1

mesh.groups = numpy.array([0.000, 0.7075, 1.415, 2.123, 2.830, 3.538, 4.245,
    5.129, 6.014, 6.898, 7.783, 8.667, 9.551, 10.44, 11.32, 12.20, 
    13.09, 1e4])*(1000/H)
mesh.dx = 0.4

mesh.I_BC = numpy.zeros((mesh.ng, 2))
mesh.F_BC = numpy.zeros((mesh.ng, 2))

# mesh.I_BC[:, 0] = 0.5*physics.group_planck(mesh, 1000/physics.K)[:, 0]
# mesh.F_BC[:, 0] = 0.5*physics.group_planck(mesh, 1000/physics.K)[:, 0]



mesh.dt = 2e-3 * 1e-8 # seconds

mesh.eps = 1e-16










# testing-- this code should be deleted




T_prev = (1/physics.K).astype(numpy.float128)*numpy.ones((mesh.nx))
kappa = group_FC_opacity(mesh, T_prev, k_star)

Cv    = FC_heatcap(1, mesh)

Q     = numpy.zeros((mesh.ng, mesh.nx))


mesh.I_BC[:, 0] = (physics.group_planck(mesh, 1000*T_prev))[:, 0]
mesh.F_BC[:, 0] = 0.5*(physics.group_planck(mesh, 1000*T_prev))[:, 0]

print(mesh.I_BC[:, 0])
print("bound")

# print(physics.group_planck(mesh, T_prev))



sol_prev = tools.Transport_solution(mesh)

sol_prev.intensity[:,:] = tools.dbl(physics.group_planck(mesh, T_prev))
# sol_prev.intensity[0:2,0:4] = 3*tools.dbl(physics.group_planck(mesh, T_prev))[0:2,0:4]

# print(sol_prev.vec[0])
# print("initial")

# print("initial condition")


coeff = tools.MG_coefficients(mesh)
coeff.assign(mesh, kappa, sol_prev, T_prev, Cv, Q)



b = tools.Transport_solution(mesh)
b.vec[:,:]  = method.unaccelerated_loop(mesh, sol_prev, T_prev, kappa, Cv, Q)

print(Cv)
print("Cv")


test = method.update_temperature(mesh, coeff, b, Cv)
print(test)
print("temp update")


T_prev = T_prev + test

# print(T_prev)
# print("new temp")

newcoeff = tools.MG_coefficients(mesh)
newcoeff.assign(mesh, kappa, b, T_prev, Cv, Q)

kappa = group_FC_opacity(mesh, T_prev, k_star)

c = tools.Transport_solution(mesh)
c.vec[:,:] = method.unaccelerated_loop(mesh, b, T_prev, kappa, Cv, Q)








plt.figure()
ax = plt.gca()
lines = tools.LD_plottable(mesh, b.vec)
tools.plot_LD_groups(mesh, lines.intensity, range(0, mesh.ng))
plt.title("b, intensity")
ax.autoscale()


plt.figure()
ax = plt.gca()
lines = tools.LD_plottable(mesh, b.vec)
tools.plot_LD_groups(mesh, lines.flux, range(0, mesh.ng))
plt.title("b, flux")
ax.autoscale()

plt.show()



def planck(T, nu):
    B = 8*numpy.pi*(H*nu)**3/((C**2 * H**2)*(numpy.exp((H*nu)/(K*T)) - 1))
    return(B)

c2 = tools.MG_coefficients(mesh)
c2.assign(mesh, kappa, sol_prev, T_prev*1000, Cv, 0)

# plt.figure()
# plt.plot(H*nu, planck(T_prev[0]*1000, nu))
# plt.stairs(c2.beta[:, 0]/numpy.diff(mesh.groups), edges = H*mesh.groups)
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1e-8, 1e20)
# plt.show()


