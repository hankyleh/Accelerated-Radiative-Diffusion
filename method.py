from physics import C
import tools
import physics # type: ignore
import numpy
import scipy.sparse as sparse
import copy
from matplotlib import pyplot as plt

def global_mat_elementwise(mesh : tools.Discretization, sigma, D):
    nx = mesh.nx + 0
    dx = mesh.dx + 0

    global_matrix = sparse.lil_array((4*nx, 4*nx))

    half = 2*nx
    
    mll = 2
    mlr = 1
    mrl = 1
    mrr = 2

    for i in range(0, nx-1):
        # left cells. Right side conditions
        I_next_R = 2*i - 1
        I_cell_L = 2*i
        I_cell_R = 2*i +1
        I_next_L = 2*i +2

        F_next_R = I_next_R + half
        F_cell_L = I_cell_L + half
        F_cell_R = I_cell_R + half
        F_next_L = I_next_L + half

        # left_row_z   = 2*i
        right_row_z = 2*i +1

        # left_row_f = left_row_z + half
        right_row_f = right_row_z + half

        # print(f"cell number {i}, right equations-- rows{right_row_z}, {right_row_f}")

        # right equation
            # Zeroth moment
        global_matrix[right_row_z, F_next_L] = 1/2
        global_matrix[right_row_z, F_cell_L] = -1/2
        global_matrix[right_row_z, I_cell_R] = 1/4
        global_matrix[right_row_z, I_next_L] = -1/4
        global_matrix[right_row_z, I_cell_L] = sigma[i]*mrl*dx/6
        global_matrix[right_row_z, I_cell_R] += sigma[i]*mrr*dx/6
            # first moment
        global_matrix[right_row_f, F_cell_L] = dx*mrl/(6)
        global_matrix[right_row_f, F_cell_R] = dx*mrr/(6)
        global_matrix[right_row_f, F_cell_R] += D[i]*3/4
        global_matrix[right_row_f, F_next_L] = -D[i]*3/4
        global_matrix[right_row_f, I_next_L] = D[i]*1/2
        global_matrix[right_row_f, I_cell_L] = -D[i]*1/2



    for i in range(1, nx):
        # right cells. Left side conditions

        
        I_next_R = 2*i - 1
        I_cell_L = 2*i
        I_cell_R = 2*i +1
        I_next_L = 2*i +2

        F_next_R = I_next_R + half
        F_cell_L = I_cell_L + half
        F_cell_R = I_cell_R + half
        F_next_L = I_next_L + half

        left_row_z   = 2*i
        # right_row_z = 2*i +1

        left_row_f = left_row_z + half
        # right_row_f = right_row_z + half

        # left equation

        # Zeroth moment
        global_matrix[left_row_z, F_cell_R] = 1/2
        global_matrix[left_row_z, F_next_R] = -1/2
        global_matrix[left_row_z, I_next_R] = -1/4
        global_matrix[left_row_z, I_cell_L] = 1/4
        global_matrix[left_row_z, I_cell_L] += sigma[i]*mll*dx/6
        global_matrix[left_row_z, I_cell_R] += sigma[i]*mlr*dx/6

        # first moment
        global_matrix[left_row_f, F_cell_L] = dx*mll/(6)
        global_matrix[left_row_f, F_cell_R] = dx*mlr/(6)
        global_matrix[left_row_f, F_next_R] = -D[i]*3/4
        global_matrix[left_row_f, F_cell_L] += D[i]*3/4
        global_matrix[left_row_f, I_cell_R] = D[i]*1/2
        global_matrix[left_row_f, I_next_R] = -D[i]*1/2


    # left boundary cell, left equation
    i = 0
    # I_next_R = 2*i - 1
    I_cell_L = 2*i
    I_cell_R = 2*i +1
    I_next_L = 2*i +2
    # F_next_R = I_next_R + half
    F_cell_L = I_cell_L + half
    F_cell_R = I_cell_R + half
    F_next_L = I_next_L + half
    left_row_z   = 2*i
    # right_row_z = 2*i +1
    left_row_f = left_row_z + half

    # Zeroth moment
    global_matrix[left_row_z, F_cell_R] = 1/2
    # global_matrix[left_row_z, F_next_R] = -1/2
    # global_matrix[left_row_z, I_next_R] = -1/4
    global_matrix[left_row_z, I_cell_L] = 1/4
    global_matrix[left_row_z, I_cell_L] += sigma[i]*mll*dx/6
    global_matrix[left_row_z, I_cell_R] = sigma[i]*mlr*dx/6

    # first moment
    global_matrix[left_row_f, F_cell_L] = dx*mll/(6)
    global_matrix[left_row_f, F_cell_R] = dx*mlr/(6)
    # global_matrix[left_row_f, F_next_R] = -D[i]*3/4
    global_matrix[left_row_f, F_cell_L] += D[i]*3/4
    global_matrix[left_row_f, I_cell_R] = D[i]*1/2
    # global_matrix[left_row_f, I_next_R] = -D[i]*1/2


    # right boundary, right equation
    i = nx-1
    I_next_R = 2*i - 1
    I_cell_L = 2*i
    I_cell_R = 2*i +1
    # I_next_L = 2*i +2
    F_next_R = I_next_R + half
    F_cell_L = I_cell_L + half
    F_cell_R = I_cell_R + half
    # F_next_L = I_next_L + half
    # left_row_z   = 2*i
    right_row_z = 2*i +1
    # left_row_f = left_row_z + half
    right_row_f = right_row_z + half

    # right equation
    # Zeroth moment
    # global_matrix[right_row_z, F_next_L] = 1/2
    global_matrix[right_row_z, F_cell_L] = -1/2
    global_matrix[right_row_z, I_cell_R] = 1/4
    # global_matrix[right_row_z, I_next_L] = -1/4
    global_matrix[right_row_z, I_cell_L] += sigma[i]*mrl*dx/6
    global_matrix[right_row_z, I_cell_R] += sigma[i]*mrr*dx/6
    # first moment
    global_matrix[right_row_f, F_cell_L] += dx*mrl/(6)
    global_matrix[right_row_f, F_cell_R] += dx*mrr/(6)
    global_matrix[right_row_f, F_cell_R] += D[i]*3/4
    # global_matrix[right_row_f, F_next_L] = -D[i]*3/4
    # global_matrix[right_row_f, I_next_L] = D[i]*1/2
    global_matrix[right_row_f, I_cell_L] += -D[i]*1/2


    global_matrix = global_matrix.tocsr()

    # print("Global Matrix assembled.")
    # print(f"Highest abs: {numpy.max(numpy.abs(global_matrix.todense()))}")
    # print(f"Lowest abs (nonzero): {numpy.min(numpy.abs(global_matrix.data))}")
    return global_matrix


# Replaced with element-wise assembly

# def assemble_global_matrix(mesh  : tools.Discretization, sigma, D):
#     # LHS of zeroth moment equation plus Fick's Law.
#     # works in the general case-- high order or low order.

#     # mesh -- variable of type "Discretization"
#     # sigma-- numpy array of size (Nx, 1)
#     # D    -- numpy array of size (Nx, 1)
#     nx = mesh.nx + 0
#     dx = mesh.dx + 0
    
#     global_matrix = sparse.lil_array((4*nx, 4*nx))
#     # interior elements
#     b_0i = tools.b_0i
#     b_0f = tools.b_0f
#     b_1i = tools.b_1i
#     b_1f = tools.b_1f
#     a  = tools.a_f 

#     M_wide = tools.M_wide

#     # interior elements
#     for i in range(1, nx-1):
#         # i = index of current interior cell

#         l_nbr = (2*i)-1
#         r_nbr = (2*i)+2
#         shft  = (2*nx)

#         # zeroth moment, intensity
#         global_matrix[2*i:(2*i + 2), l_nbr:r_nbr+1] += b_0i + (sigma[i]*dx*M_wide)

#         # zeroth moment, flux
#         global_matrix[2*i:2*i + 2, l_nbr+shft:r_nbr+shft+1] += a + b_0f

#         # first moment, intensity
#         global_matrix[2*i +shft:2*i + 2+shft, l_nbr:r_nbr+1] += D[i] * (b_1i + a) 

#         # first moment, flux
#         global_matrix[2*i+shft:2*i+2+shft,l_nbr+shft:r_nbr+1+shft] \
#             += (dx*M_wide) + (D[i]*b_1f)

#     # boundary elements
#     # -----------------
#     # Left boundary
#     # zeroth, intensity
#     global_matrix[0:2, 0:3] += b_0i[:, 1:] + (sigma[0]*dx*M_wide[:, 1:])
#     # zeroth, flux
#     global_matrix[0:2, shft:3+shft] += a[:, 1:] + b_0f[:, 1:]
#     # first, intensity
#     global_matrix[shft:2+shft, 0:3] += D[0] * (b_1i[:, 1:] + a[:, 1:]) 
#     # first, flux
#     global_matrix[shft:2+shft,shft:3+shft] += ((dx*M_wide[:, 1:]) + 
#                                                (D[0]*b_1f[:, 1:]))
#     # Right boundary
#     # zeroth, intensity
#     global_matrix[shft-2:shft, shft-3:shft] += b_0i[:, 0:-1] + (sigma[-1]*dx*M_wide[:, 0:-1])
#     # zeroth, flux
#     global_matrix[shft-2:shft, -3:] += a[:, 0:-1] + b_0f[:, 0:-1] 
#     # first, intensity
#     global_matrix[-2:, shft-3:shft] += D[-1] * (b_1i[:, 0:-1] + a[:, 0:-1]) 
#     # first, flux
#     global_matrix[-2:, -3:] += ((dx*M_wide[:, 0:-1]) + 
#                                                (D[-1]*b_1f[:, 0:-1]))

#     global_matrix = global_matrix.tocsr()
#     return global_matrix

def get_HO_source(
        mesh  : tools.Discretization, 
        prev_I, 
        coeff : tools.MG_coefficients, 
        k     : int):
    source = numpy.zeros((4*mesh.nx))

    # compute fission source, add to 'S'
    dx = mesh.dx
    nx = mesh.nx
    M = tools.M

    mass_global = sparse.lil_array((2*nx, 2*nx))

    for i in range(0, nx):
        mass_global[2*i : 2*i+2, 2*i : 2*i+2] += M[:]
    mass_global[:] = mass_global.tocsr()

    fiss = numpy.zeros((mesh.ng, 2*nx))


   
    for g in range(0, mesh.ng):
        fiss[g] += tools.dbl(coeff.sig_f[g])*(prev_I[g])

        
    source[0:2*mesh.nx] += (dx*( mass_global @ coeff.S[k])
                + (tools.dbl(coeff.eta) * tools.dbl(coeff.chi[k])
                * dx* (mass_global @ numpy.sum(fiss, axis=0))))
    # add BCs
    source[0] += mesh.F_BC[k, 0]
    source[(2*mesh.nx)-1] += -mesh.F_BC[k, 1]




    source[2*mesh.nx] += mesh.I_BC[k, 0]
    source[-1] +=        -mesh.I_BC[k, 1]


    return source

def get_LO_source(mesh : tools.Discretization, coeff : tools.Grey_coeff):
    nx =mesh.nx
    M = tools.M

    mass_global = sparse.lil_array((2*nx, 2*nx))

    for i in range(0, nx):
        mass_global[2*i : 2*i+2, 2*i : 2*i+2] += M[:]
    mass_global[:] = mass_global.tocsr()

    source = numpy.zeros((4*mesh.nx))
    source[:2*nx] = tools.dbl(coeff.eta * mesh.dx) * (mass_global @ coeff.r)

    return source


def assemble_HO(mesh : tools.Discretization, coeff : tools.MG_coefficients, last_iter_I, k):
    system = tools.Global_system()
    system.mat = global_mat_elementwise(mesh, coeff.sig_a+coeff.sig_f[k], coeff.D[k])

    system.src = get_HO_source(mesh, last_iter_I, coeff, k)

    return system


def assemble_LO(mesh : tools.Discretization, coeff : tools.Grey_coeff):
    system = tools.Global_system()
    system.mat = global_mat_elementwise(mesh, coeff.sig_a + (1-coeff.eta)*coeff.sigf_avg, coeff.D_avg)

    system.src = get_LO_source(mesh, coeff)
    return system

def unaccelerated_loop(mesh : tools.Discretization, 
                       sol_prev : tools.Transport_solution, 
                       T_prev, 
                       kappa, 
                       Cv, 
                       Q):
    print(T_prev)
    print("temperature")

    coeff = tools.MG_coefficients(mesh)
    I_prev = sol_prev.intensity[:]
    coeff.assign(mesh, kappa, sol_prev, T_prev, Cv, Q)

    last_iteration = tools.Transport_solution(mesh)
    updated_solution =  tools.Transport_solution(mesh)
    last_iteration.vec[:,:] = sol_prev.vec[:].copy()
    
    change = [1]
    iter = 0

    while (numpy.max(change) > mesh.eps) :
        iter += 1
        print(f"Iteration {iter}, change = {numpy.max(change)}")

        for k in range(0, mesh.ng):
            sys = assemble_HO(mesh, coeff, last_iteration.intensity, k)
            updated_solution.vec[k, :], b = sparse.linalg.gmres(sys.mat, sys.src, x0=last_iteration.vec[k], atol = mesh.eps)

        last_iteration.vec[:] = copy.deepcopy(updated_solution.vec[:])
        diff = abs((updated_solution.intensity / last_iteration.intensity)- 1) 
        
        change = numpy.linalg.norm(diff, 2, axis=1)
    return updated_solution.vec


def accelerated_loop(mesh : tools.Discretization, 
                       sol_prev : tools.Transport_solution, 
                       T_prev, 
                       kappa, 
                       Cv, 
                       Q):
    # TODO


    coeff = tools.MG_coefficients(mesh)
    I_prev = sol_prev.intensity[:]
    coeff.assign(mesh, kappa, sol_prev, T_prev, Cv, Q)

    last_iteration   = tools.Transport_solution(mesh)
    updated_solution =  tools.Transport_solution(mesh)
    error_soln = tools.Transport_solution(mesh)


    last_iteration.vec[:,:] = copy.deepcopy(sol_prev.vec)

    grey_constants = tools.Grey_coeff(mesh)
    




    change = [1]
    iter = 0

    while (numpy.max(change) > mesh.eps) :
        iter += 1
        print(f"Iteration {iter}, change = {numpy.max(change)}")

        for k in range(0, mesh.ng):
            sys = assemble_HO(mesh, coeff, last_iteration.intensity, k)
            updated_solution.vec[k, :], b = sparse.linalg.gmres(sys.mat, sys.src, atol=mesh.eps, rtol = mesh.eps, x0=last_iteration.vec[k])

        grey_constants.assign(mesh, coeff, updated_solution, last_iteration)


        sys_grey = assemble_LO(mesh, grey_constants)
        # error_soln.vec[:] , b = sparse.linalg.lgmres(sys_grey.mat, sys_grey.src, atol=mesh.eps, rtol = mesh.eps)
        error_soln.vec[:] = sparse.linalg.inv(sys_grey.mat) @ sys_grey.src

        # plt.figure()
        # plt.matshow(sys_grey.mat.todense())
        # plt.show()

        print("error solution found, ")
        print(error_soln.vec[:])

        print("error energy spectrum")
        print(grey_constants.spectrum[:,:])

        last_iteration.vec[:] = updated_solution.vec[:] + error_soln.vec[:]*numpy.tile(tools.dbl(grey_constants.spectrum), reps = (1, 2))
        diff = abs((updated_solution.intensity / last_iteration.intensity)- 1) 
        
        change = numpy.linalg.norm(diff, 2, axis=1)
    return updated_solution.vec
    return 0

def update_temperature(mesh : tools.Discretization, 
                       coeff : tools.MG_coefficients, 
                       soln : tools.Transport_solution,
                       Cv,
                       Q = 0):
    # TODO

    # print(soln.cell_center_i[0])
    # print("intensity")
    # print(coeff.beta[0])
    # print("beta")
    # print(coeff.kappa[0] * (soln.cell_center_i[0] - coeff.beta[0]))
    # print("diff")

    # print((Cv/mesh.dt))
    # print("Cv/dt")

    # print(numpy.sum(coeff.kappa * coeff.db_dt, axis = 0))
    # print("kappa * dbdt")
    temp = (
        numpy.sum(coeff.kappa * (soln.cell_center_i - coeff.beta), axis=0) + Q
    )/(
        (Cv/mesh.dt) + numpy.sum(coeff.kappa * coeff.db_dt, axis = 0)
    )

    return temp

    # calculate planck function, dbdt, all necessary constants

def solve_unaccelerated(mesh : tools.Discretization, 
                        scale : tools.Scales, opacity, 
                        IC : tools.Transport_solution, 
                        T_init, 
                        Cv, 
                        Q=0):
    print("Beginning unaccelerated iteration")

    dt = mesh.dt
    nu = mesh.groups

    nt = mesh.nt
    k_star = 27

    # gets intensity, flux, and temperature at each provided time stop 

    transport_output = []
    temp_output = []
    transport_iters = tools.Transport_solution(mesh)
    sol_prev = tools.Transport_solution(mesh)

    sol_prev.vec[:] = copy.deepcopy(IC.vec)

    T_iter = copy.deepcopy(T_init)

    kappa = numpy.zeros((mesh.ng, mesh.nx))
    change = numpy.zeros(mesh.nx)

    for stop in range(0, len(nt)):

        print(f"Starting steps towards {nt[stop]}")


        for step in range(0, nt[stop]):

            kappa[:] = opacity(mesh, T_iter, k_star)

            transport_iters.vec[:,:]  = unaccelerated_loop(mesh, sol_prev, T_iter, kappa, Cv, Q)
            sol_prev.vec[:] = transport_iters.vec[:].copy()
            coeff = tools.MG_coefficients(mesh)
            coeff.assign(mesh, kappa, transport_iters, T_iter, Cv, Q)

            change[:] = update_temperature(mesh, coeff, transport_iters, Cv)
            T_iter += copy.deepcopy(change[:])
            coeff.assign(mesh, kappa, transport_iters, T_iter, Cv, Q)
        transport_output.append(transport_iters)
        temp_output.append(temp_output)
        
    return temp_output, transport_output


def solve_accelerated(mesh : tools.Discretization, 
                        scale : tools.Scales, opacity, 
                        IC : tools.Transport_solution, 
                        T_init, 
                        Cv, 
                        Q=0):
    print("Beginning accelerated iteration")

    dt = mesh.dt
    nu = mesh.groups

    nt = mesh.nt
    k_star = 27

    # gets intensity, flux, and temperature at each provided time stop 

    transport_output = []
    temp_output = []
    transport_iters = tools.Transport_solution(mesh)
    sol_prev = tools.Transport_solution(mesh)

    sol_prev.vec[:] = copy.deepcopy(IC.vec)

    T_iter = copy.deepcopy(T_init)

    kappa = numpy.zeros((mesh.ng, mesh.nx))
    change = numpy.zeros(mesh.nx)

    # for stop in range(0, len(nt)):

    #     print(f"Starting steps towards {nt[stop]}")


    #     for step in range(0, nt[stop]):

    #         kappa[:] = opacity(mesh, T_iter, k_star)

    # ACCLERATED LOOP!!!!!!!

    #         transport_iters.vec[:,:]  = accelerated_loop(mesh, sol_prev, T_iter, kappa, Cv, Q)
    #         sol_prev.vec[:] = transport_iters.vec[:].copy()
    #         coeff = tools.MG_coefficients(mesh)
    #         coeff.assign(mesh, kappa, transport_iters, T_iter, Cv, Q)

    #         change[:] = update_temperature(mesh, coeff, transport_iters, Cv)
    #         T_iter += copy.deepcopy(change[:])
    #         coeff.assign(mesh, kappa, transport_iters, T_iter, Cv, Q)
    #     transport_output.append(transport_iters)
    #     temp_output.append(temp_output)
        
    # return temp_output, transport_output



    
        
    
