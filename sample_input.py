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



def FC_opacity(T, nu, k0):
    def num_func(n):
        return (n**7)*(1-numpy.exp(-physics.H*n/(physics.K*T)))**-3
    def dem_func(n):
        return (n**4)*(1-numpy.exp(-physics.H*n/(physics.K*T)))**-2
    
    

    numerator, err = integrate.quad(num_func, nu[0], nu[1])
    denominator, err = integrate.quad(dem_func, nu[0], nu[1])

    if numerator == 0:
        print(T)
        print(nu)
        print(numerator)
        print(denominator)

    return (1000**3)*k0 * denominator/(numerator*((physics.H)**3))

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



# test_kappa = FC_opacity(1/physics.K, [2000, 2002]/physics.H, 27)
# print(test_kappa)

# plt.figure()
# plt.show()




scale = tools.Scales()
mesh = tools.Discretization()

k_star = 27

# scale.I = 1

mesh.groups = numpy.array([0.0000001, 0.3, 0.6, 0.8, 1.2, 1.5, 1.8, 2.4, 
                           2.7, 3, 4, 5, 7, 9, 11, 15, 20, 1e4])*(1000/physics.H)
mesh.dx = 0.4

mesh.t_stops = numpy.array([0, 2e-2, 5e-2]) * 1e-8

mesh.I_BC = numpy.zeros((mesh.ng, 2))
mesh.F_BC = numpy.zeros((mesh.ng, 2))

# mesh.I_BC[:, 0] = 0.5*physics.group_planck(mesh, 1000/physics.K)[:, 0]
# mesh.F_BC[:, 0] = 0.5*physics.group_planck(mesh, 1000/physics.K)[:, 0]



mesh.dt = 2e-1 * 1e-8 # seconds


mesh.eps = 1e-15



print(mesh.groups)
print("groups, hz")

print(mesh.groups * physics.H)
print("groups, eV")




# plt.figure()
# plt.show()






# testing-- this code should be deleted




T_prev = (1/physics.K).astype(numpy.float128)*numpy.ones((mesh.nx))
T_bound = (1000/physics.K).astype(numpy.float128)*numpy.ones((mesh.nx))
kappa = group_FC_opacity(mesh, T_prev, k_star)


Cv    = FC_heatcap(1/physics.K, mesh)


print(Cv)
print("Cv")
print("NTOE Cv has been modified-- check back later")

Q     = numpy.zeros((mesh.ng, mesh.nx))

mesh.I_BC[:, 0] = (physics.group_planck(mesh, T_bound))[:, 0]
mesh.F_BC[:, 0] = 0.5*(physics.group_planck(mesh, T_bound))[:, 0]

print(mesh.I_BC[:, 0])
print("bound")

# print(physics.group_planck(mesh, T_prev))

kappa = group_FC_opacity(mesh, T_prev, k_star)
coeff = tools.MG_coefficients(mesh)


sol_prev = tools.Transport_solution(mesh)
sol_prev.intensity[:,:] = tools.dbl(physics.group_planck(mesh, T_prev))
print(sol_prev.intensity[:,0:6])

coeff.assign(mesh, kappa, sol_prev, T_prev, Cv, Q)

# print(kappa[:,0])
# print("kappa original")
# print(coeff.kappa[:,0])
# print("coeff kappa")
# print(coeff.D[:, 0])
# print("D, coeff")


plt.figure()
ax = plt.gca()
lines = tools.LD_plottable(mesh, sol_prev.vec)
tools.plot_LD_groups(mesh, lines.intensity, [0])
plt.autoscale()


# plt.show()


# plt.figure()

change = T_prev.copy()
b = tools.Transport_solution(mesh)

solutions = []
solutions.append(copy.deepcopy(b))

# plt.plot(mesh.cell_centers, T_prev*physics.K, label=f"{0:.2e}")
# plt.title("Temperature, initial")




temp_vec = []

sol_test = tools.Transport_solution(mesh)

plt.figure()
ax = plt.gca()
lines = tools.LD_plottable(mesh, sol_prev.vec)
tools.plot_LD_groups(mesh, lines.intensity, range(0, mesh.ng))
plt.title("previous solution, kappa=1")
plt.autoscale()


kappa_test = 1000*numpy.ones((mesh.nx))

coeff = tools.MG_coefficients(mesh)
coeff.assign(mesh, kappa_test, sol_prev, T_prev, Cv, Q)


mat_test = method.global_mat_elementwise(mesh, kappa_test, 1/(3*kappa_test))
src_test = method.get_HO_source(mesh, sol_prev.intensity, coeff, 0)

sol_test.vec[:] = sparse.linalg.inv(mat_test) @ src_test

mat_test = method.global_mat_elementwise(mesh, kappa_test, 1/(3*kappa_test))
src_test = method.get_HO_source(mesh, sol_prev.intensity, coeff, 0)

sol_test.vec[:] = sparse.linalg.inv(mat_test) @ src_test

plt.figure()
ax = plt.gca()
lines = tools.LD_plottable(mesh, sol_test.vec)
tools.plot_LD_groups(mesh, lines.intensity, [0])
plt.title(f"Test intensity, kappa={kappa_test[0]} cm-1")
plt.autoscale()
plt.savefig(f"kappa{kappa_test[0]}_intensity.png")
plt.close()

plt.figure()
ax = plt.gca()
lines = tools.LD_plottable(mesh, sol_test.vec)
tools.plot_LD_groups(mesh, lines.flux, range(0, mesh.ng))
plt.title(f"Test flux, kappa={kappa_test[0]} cm-1")
plt.autoscale()
plt.savefig(f"kappa{kappa_test[0]}_flux.png")
plt.close()


# plt.show()




# a = method.unaccelerated_loop(mesh, sol_test, T_prev, kappa, Cv, Q)

# plt.figure()
# plt.show()




# plt.figure()

# for t in range(0, len(temp_vec)):
#     plt.plot(mesh.cell_centers, temp_output[-1]*physics.K, label=f"{(t+1)*mesh.dt:.2e}")
# plt.legend()
# plt.title("Temperature, final")

    
# plt.figure()
# ax = plt.gca()
# lines = tools.LD_plottable(mesh, transport_output[-1].vec)
# tools.plot_LD_groups(mesh, lines.intensity, range(0, mesh.ng))
# plt.title("b, intensity")
# ax.autoscale()



# plt.figure()
# ax = plt.gca()
# lines = tools.LD_plottable(mesh, b.vec)
# tools.plot_LD_groups(mesh, lines.flux, range(0, mesh.ng))
# plt.title("b, flux")
# ax.autoscale()

# plt.show()


# plt.figure()
# ax = plt.gca()

# for i in range(0, len(solutions)):
#     lines = tools.LD_plottable(mesh, solutions[i].vec)
#     tools.plot_LD_groups(mesh, lines.intensity, [0])
# plt.title("intensity over time")
# ax.autoscale()




# plt.show()

# print(coeff.kappa[:])
# print("kappa")



