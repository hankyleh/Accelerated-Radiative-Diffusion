import numpy
from constant import H, K


def cumulative_planck(T, nu):
    # evaluates integral(B) d nu, from 0 to nu
    z = (H*nu)/(K*T)
    sigma =( 
        (z <= 2) * 
            ((z**3) * ((1/3) - (z/8) + ((z**2)/62.4))) +
        (z > 2)  * 
            (6.4939 - (numpy.exp(-z) * ((z**3)+(3*(z**2))+(6*z)+7.28))) 
    )

    return sigma

def group_planck(T, groups):
    # evaluates integral(B) d nu, along each group bound
    return numpy.diff(cumulative_planck(T, groups))

def cumulative_dB_dT(T, nu):
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
        
def group_dB_dT(T, groups):
    return numpy.diff(cumulative_dB_dT(T, groups))