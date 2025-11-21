# import numpy
# import matplotlib



import numpy
import scipy.integrate as integrate
import constant # type: ignore
from constant import H, K, A_R # type: ignore
import method # type: ignore
import physics # type: ignore




# Define temperature-dependent coefficients as functions

def FC_opacity(T, nu, k0):
    def num_func(n):
        return (H**3)*(n**7)*(1-numpy.exp(-H*n/(K*T)))**-2
    def dem_func(n):
        return (n**4)*(1-numpy.exp(-H*n/(K*T)))**-2

    numerator, err = integrate.quad(num_func, nu[0], nu[1])
    denominator, err = integrate.quad(dem_func, nu[0], nu[1])
    return k0 * denominator/numerator

def FC_heatcap(T_b, mesh):
    return (0.5917*A_R*(T_b)**3) * numpy.ones(mesh.cell_centers.size)





BC_left = 100
BC_right = 0

initial_I = 0 # Initial 
initial_T = 0 # Initial temperature array

# Units for scaling computations, such that
# x_physical = x_working * scale_x

scale = constant.scales()
mesh = constant.discretization()



scale.I = 2


mesh.groups = numpy.array([0.000, 0.3, 0.6 ,0.8, 1.2, 1.5, 1.8, 2.4, 2.7, 3.0, 4.0,
                       5.0, 7.0, 9.0, 11.0, 15.0, 20.0, 1e10])*(1000/H)
mesh.dx = 0.1
# mesh.init()






# testing-- this code should be deleted



T = numpy.array([1, 10**2, 10**3])/K.astype(numpy.float128)

t = 2
g = 3
ans = FC_opacity(T[t], mesh.groups[g:g+2], 27)
# print(ans)


print (physics.group_planck(T[t], mesh.groups))
print (physics.group_dB_dT(T[t], mesh.groups))

print(mesh.dx)
print(mesh.cell_edges)
print(mesh.cell_centers)


# print(scale.B)
# print(method.func())


