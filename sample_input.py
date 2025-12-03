# import numpy
# import matplotlib



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





BC_left = 100
BC_right = 0

initial_I = 0 # Initial 
initial_T = 0 # Initial temperature array

# Units for scaling computations, such that
# x_physical = x_working * scale_x

scale = tools.Scales()
mesh = tools.Discretization()

k_star = 27



scale.I = 1


mesh.groups = numpy.array([0.000000, 0.7075, 1.415, 2.123, 2.830, 3.538, 4.245,
    5.129, 6.014, 6.898, 7.783, 8.667, 9.551, 10.44, 11.32, 12.20, 
    13.09, 1e4])*(1000/H)
mesh.dx = 0.4

mesh.I_BC = numpy.zeros((mesh.ng, 2))
mesh.F_BC = numpy.zeros((mesh.ng, 2))

mesh.I_BC[:, 0] = physics.group_planck(mesh, 1000/physics.K)[:, 0]
mesh.F_BC[:, 0] = physics.group_planck(mesh, 1000/physics.K)[:, 0]

# print(mesh.I_BC)
# print(mesh.F_BC)
mesh.dt = 2e-13
# mesh.I_BC[-4, 0] = 100
# mesh.F_BC[-4, 0] = 100


mesh.eps = 1e-6
# print(mesh.cell_centers)
# print(mesh.cell_edges)







# testing-- this code should be deleted



# out = method.assemble_global_matrix(mesh, (0*mesh.cell_centers)+1, (0*mesh.cell_centers)+1)
# print (out.mat.todense())

# plt.figure()
# plt.spy(out.todense())
# # plt.savefig("fig")
# plt.show()
# plt.close()

T_prev = (1/physics.K).astype(numpy.float128)*numpy.ones((mesh.nx))
kappa = group_FC_opacity(mesh, T_prev,  k_star)
# print(kappa)
# kappa = numpy.ones((mesh.ng, mesh.nx))
Cv    = numpy.tile(FC_heatcap(1, mesh), reps=(mesh.ng, 1))
# print(Cv)
# print("heat capacity")
Q     = numpy.zeros((mesh.ng, mesh.nx))



# print(physics.group_planck(mesh, T_prev))


sol_prev = tools.Transport_solution(mesh)
b = tools.Transport_solution(mesh)
sol_prev.intensity = 2*tools.dbl(physics.group_planck(mesh, T_prev))

coeff = tools.MG_coefficients(mesh)
coeff.assign(mesh, kappa, sol_prev, T_prev, Cv, Q)

# print(coeff.S)
# print(coeff.eta)
# print(coeff.chi)

a = method.get_HO_source(mesh, sol_prev.intensity, coeff, mesh.ng-4)
# print(coeff.D)
# print(a)

b.vec  = method.unaccelerated_loop(mesh, sol_prev, T_prev, kappa, Cv, Q)


# print(coeff.beta)
# print("beta")
# print(coeff.db_dt)
# print("dbdt")
# print(b.cell_center_i)
# print("cell center I")

test = method.update_temperature(mesh, coeff, b, T_prev, Cv)
print(test)

# print(b)

# print(b.intensity)
# print(b.cell_center_i)


# print("intensity")

# plt.figure()
# ax = plt.gca()
# lines = tools.LD_plottable(mesh, sol_prev.vec)
# tools.plot_LD_groups(mesh, lines.intensity, range(0, mesh.ng))
# plt.title("Initial state, intensity")
# ax.autoscale()
# # plt.show()

# center = b.cell_center_i
# print(mesh.cell_centers.shape)
# print(center.shape)

# plt.figure()
# ax = plt.gca()
# lines = tools.LD_plottable(mesh, b.vec)
# # tools.plot_LD_groups(mesh, lines.intensity, range(0, mesh.ng))
# tools.plot_LD_groups(mesh, lines.intensity, range(0, mesh.ng))
# plt.title("time-advanced state, intensity")
# ax.autoscale()

# plt.show()




# def planck(T, nu):
#     B = 2*H*(nu**3)/((C**2)*(numpy.exp((H*nu)/(K*T)) - 1))
#     return(B)

# vec = numpy.linspace(0.001, 1e4, 1000)
# log_nu = numpy.linspace(numpy.log(0.001*mesh.groups[1]), numpy.log(mesh.groups[-1]), 250)
# nu = numpy.exp(log_nu)

# plt.figure()
# # plt.plot(H*nu, planck(T_prev[0], nu))
# plt.stairs(kappa[:, 0], edges = H*mesh.groups)
# plt.xscale('log')
# plt.yscale('log')
# # plt.ylim(1e-8, 1e20)
# plt.show()





# sample = numpy.zeros((mesh.ng, mesh.nx*4))
# sample[0,::2] = 1
# sample[1,::2] = 2
# sample[2,1::2] = 2
# sample[2,::2] = 3
# sample[3,1::2] = 3
# sample[3,::2] = 4
# print(sample)

# fig = plt.figure()
# ax = plt.gca()
# lines = constant.LD_plottable(mesh, sample)
# legend_proxies = []
# legend_labels = []

# for g in range(0, mesh.ng):
#     c = lines.flux[g]
#     c.set_color(constant.colors[g])
#     ax.add_collection(c)

#     legend_proxies.append(matplotlib.lines.Line2D([0],[0], color=constant.colors[g], linestyle = "-"))
#     legend_labels.append(g)
# ax.autoscale()
# plt.legend(legend_proxies, legend_labels)
# plt.show()



# T = numpy.array([1, 10**2, 10**3])/K.astype(numpy.float128)

# t = 2
# g = 3
# ans = FC_opacity(T[t], mesh.groups[g:g+2], 27)
# # print(ans)


# print (physics.group_planck(mesh, T[t], mesh.groups))
# print (physics.group_dB_dT(mesh, T[t], mesh.groups))

# print(mesh.dx)
# print(mesh.cell_edges)
# print(mesh.cell_centers)


# print(scale.B)
# print(method.func())


