import numpy
import tools

H = numpy.float128(4.135667696e-15) # eV * s
K = numpy.float128(8.617333262e-5)  # eV / K
C = numpy.float128(2.99792458e10)   # cm / s
SIG_R = numpy.float128(3.53916934e7) # eV / (cm^2 * s * K^4)
A_R = numpy.float128(4.72215928e-3)  # eV / (cm^2 * s * K^4)


def cumulative_sigma(mesh, T, nu):
    # evaluates integral(B) d nu, from 0 to nu
    z = (H*nu)/(K*T)
    sigma =( 
        (z <= 2) * 
            ((z**3) * ((1/3) - (z/8) + ((z**2)/62.4))) +
        (z > 2)  * 
            (6.4939 - (numpy.exp(-z) * ((z**3)+(3*(z**2))+(6*z)+7.28))) 
    )

    return sigma

def group_planck(mesh, T):
    # evaluates integral(B) d nu, along each group bound
    groups = mesh.groups[:, numpy.newaxis].reshape((mesh.ng+1, 1))

    if numpy.size(T) != 1:
        T.reshape((1, mesh.nx))
        t = numpy.tile(T, (mesh.ng+1, 1))
    else:
        t = numpy.tile(T, (mesh.ng+1, mesh.nx))
    g = numpy.tile(groups, (1, mesh.nx))
    
    return (2*(K*t[1:, :])**4)* numpy.diff(cumulative_sigma(mesh, t, g), axis=0)/((H**3)*(C**2))

def cumulative_dsigma_dT(mesh, T, nu):
    z = (H*nu)/(K*T)
    d = ( 
        (z <= 2) * 
            (((3*z**2) * ((1/3) - (z/8) + ((z**2)/62.4)))
             + ((z**3) * ((-1/8) + ((2*z)/62.4)))) +
        (z > 2)  * 
            ((numpy.exp(-z) * ((z**3)+(3*(z**2))+(6*z)+7.28)) -
             (numpy.exp(-z) * ((3*z**2)+(6*z)+(6)))) 
    )
    return d
        
def group_dB_dT(mesh, T):
    
    groups = mesh.groups[:, numpy.newaxis].reshape((mesh.ng+1, 1))
    T.reshape((1, mesh.nx))
    g = numpy.tile(groups, (1, mesh.nx))
    t = numpy.tile(T, (mesh.ng+1, 1))

    return (
        (2*(K*t[1:, :])**4)*numpy.diff(cumulative_dsigma_dT(mesh, t, g), axis=0)
            /((H**3)*(C**2))) + (
                    (8*K*(K*t[1:, :])**3)*
                numpy.diff(cumulative_sigma(mesh, t, g), axis=0))