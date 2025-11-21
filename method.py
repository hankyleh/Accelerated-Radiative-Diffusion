from constant import C
import constant # type: ignore
import physics # type: ignore
import numpy



def high_order_sweep(mesh, kappa, I_prev, T_prev, Cv, Q):
    sig_a = 1/(C*mesh.dt)
    sig_f = kappa
    s = q + I_prev/(C*mesh.dt)
    db_dt_g = physics.group_dB_dT(T_prev, mesh.groups)
    Bg = physics.group_planck(T_prev, mesh.groups)
    eta = (numpy.sum(kappa * db_dt_g)
                /
                ((Cv / mesh.dt)+
                    numpy.sum(kappa * db_dt_g))
            )
    chi = (kappa * db_dt_g) / numpy.sum(kappa * db_dt_g)
    q   = ((kappa * Bg) + 
           eta*chi*(Q - numpy.sum(kappa * Bg)))

    I = numpy.zeros(I_prev.size)
    # TODO
    # set up matrix + source vector
    # pcg or other iterative solve
    return I

def get_grey_constants():
    # TODO
    # calculate grey diffusion constants
    return 0

def low_order_sweep():
    # TODO
    # set up matrix + source vector
    # pcg or other iterative solve
    return 0

def unaccelerated_loop():
    # TODO
    # while change > tol, do:
        # high_order_sweep
        # calculate change
    return 0

def accelerated_loop():
    # TODO
    # while change > tol, do:
        # high_order_sweep
        # get_grey_constants
        # low_order_sweep
        # calculate change
    return 0

def solve(mesh, scale, opacity):
    # TODO
    dt = mesh.dt
    nu = mesh.groups
    # scale all inputs
    # for each time step:
        # compute temperature-dependent coefficients using previous temperature
        # (un)accelerated_loop() based on specified method
        # calculate temperature change
        # populate time-dependent temperature & intensity vector at this time
    # rescale for outputs
    return 0