from physics import C
import tools
import physics # type: ignore
import numpy
import scipy.sparse as sparse
import copy


def global_mat_elementwise(mesh : tools.Discretization, sigma, D):

    nx = mesh.nx + 0
    dx = mesh.dx + 0

    global_matrix = sparse.lil_array((4*nx, 4*nx))

    half = 2*nx
    

    mll = 2
    mlr = 1
    mrl = 1
    mrr = 2

    # print(f"D = {D[0]}")


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
        global_matrix[right_row_f, F_cell_L] = dx*mrl/(6*D[i])
        global_matrix[right_row_f, F_cell_R] = dx*mrr/(6*D[i])
        global_matrix[right_row_f, F_cell_R] += 3/4
        global_matrix[right_row_f, F_next_L] = -3/4
        global_matrix[right_row_f, I_next_L] = 1/2
        global_matrix[right_row_f, I_cell_L] = -1/2



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


        # print(f"cell number {i}, left equations. rows {left_row_f}, {left_row_z}")

        # left equation

        # Zeroth moment
        global_matrix[left_row_z, F_cell_R] = 1/2
        global_matrix[left_row_z, F_next_R] = -1/2
        global_matrix[left_row_z, I_next_R] = -1/4
        global_matrix[left_row_z, I_cell_L] = 1/4
        global_matrix[left_row_z, I_cell_L] += sigma[i]*mll*dx/6
        global_matrix[left_row_z, I_cell_R] += sigma[i]*mlr*dx/6

        # first moment
        global_matrix[left_row_f, F_cell_L] = dx*mll/(6*D[i])
        global_matrix[left_row_f, F_cell_R] = dx*mlr/(6*D[i])
        global_matrix[left_row_f, F_next_R] = -3/4
        global_matrix[left_row_f, F_cell_L] += 3/4
        global_matrix[left_row_f, I_cell_R] = 1/2
        global_matrix[left_row_f, I_next_R] = -1/2


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
    global_matrix[left_row_f, F_cell_L] = dx*mll/(6*D[i])
    global_matrix[left_row_f, F_cell_R] = dx*mlr/(6*D[i])
    # global_matrix[left_row_f, F_next_R] = -D[i]*3/4
    global_matrix[left_row_f, F_cell_L] += 3/4
    global_matrix[left_row_f, I_cell_R] = 1/2
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
    global_matrix[right_row_z, I_cell_L] = sigma[i]*mrl*dx/6
    global_matrix[right_row_z, I_cell_R] += sigma[i]*mrr*dx/6
    # first moment
    global_matrix[right_row_f, F_cell_L] = dx*mrl/(6*D[i])
    global_matrix[right_row_f, F_cell_R] = dx*mrr/(6*D[i])
    global_matrix[right_row_f, F_cell_R] += 3/4
    # global_matrix[right_row_f, F_next_L] = -D[i]*3/4
    # global_matrix[right_row_f, I_next_L] = D[i]*1/2
    global_matrix[right_row_f, I_cell_L] = -1/2


    global_matrix = global_matrix.tocsr()

    # print("Global Matrix assembled.")
    # print(f"Highest abs: {numpy.max(numpy.abs(global_matrix.todense()))}")
    # print(f"Lowest abs (nonzero): {numpy.min(numpy.abs(global_matrix.data))}")
    return global_matrix




def assemble_global_matrix(mesh  : tools.Discretization, sigma, D):
    # LHS of zeroth moment equation plus Fick's Law.
    # works in the general case-- high order or low order.

    # mesh -- variable of type "Discretization"
    # sigma-- numpy array of size (Nx, 1)
    # D    -- numpy array of size (Nx, 1)
    nx = mesh.nx + 0
    dx = mesh.dx + 0
    
    global_matrix = sparse.lil_array((4*nx, 4*nx))
    # interior elements
    b_0i = tools.b_0i
    b_0f = tools.b_0f
    b_1i = tools.b_1i
    b_1f = tools.b_1f
    a  = tools.a_f 

    M_wide = tools.M_wide

    # interior elements
    for i in range(1, nx-1):
        # i = index of current interior cell

        l_nbr = (2*i)-1
        r_nbr = (2*i)+2
        shft  = (2*nx)

        # zeroth moment, intensity
        global_matrix[2*i:(2*i + 2), l_nbr:r_nbr+1] += b_0i + (sigma[i]*dx*M_wide)

        # zeroth moment, flux
        global_matrix[2*i:2*i + 2, l_nbr+shft:r_nbr+shft+1] += a + b_0f

        # first moment, intensity
        global_matrix[2*i +shft:2*i + 2+shft, l_nbr:r_nbr+1] += D[i] * (b_1i + a) 

        # first moment, flux
        global_matrix[2*i+shft:2*i+2+shft,l_nbr+shft:r_nbr+1+shft] \
            += (dx*M_wide) + (D[i]*b_1f)

    # boundary elements
    # -----------------
    # Left boundary
    # zeroth, intensity
    global_matrix[0:2, 0:3] += b_0i[:, 1:] + (sigma[0]*dx*M_wide[:, 1:])
    # zeroth, flux
    global_matrix[0:2, shft:3+shft] += a[:, 1:] + b_0f[:, 1:]
    # first, intensity
    global_matrix[shft:2+shft, 0:3] += D[0] * (b_1i[:, 1:] + a[:, 1:]) 
    # first, flux
    global_matrix[shft:2+shft,shft:3+shft] += ((dx*M_wide[:, 1:]) + 
                                               (D[0]*b_1f[:, 1:]))
    # Right boundary
    # zeroth, intensity
    global_matrix[shft-2:shft, shft-3:shft] += b_0i[:, 0:-1] + (sigma[-1]*dx*M_wide[:, 0:-1])
    # zeroth, flux
    global_matrix[shft-2:shft, -3:] += a[:, 0:-1] + b_0f[:, 0:-1] 
    # first, intensity
    global_matrix[-2:, shft-3:shft] += D[-1] * (b_1i[:, 0:-1] + a[:, 0:-1]) 
    # first, flux
    global_matrix[-2:, -3:] += ((dx*M_wide[:, 0:-1]) + 
                                               (D[-1]*b_1f[:, 0:-1]))

    global_matrix = global_matrix.tocsr()
    return global_matrix

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
        mass_global[2*i : 2*i+2, 2*i : 2*i+2] = M
    mass_global = mass_global.tocsr()

    fiss = numpy.zeros((mesh.ng, 2*nx))

    # print(k)
    # print("k")


    # print(coeff.S[k])
    # print("S")

    # print(coeff.q[k])
    # print("q")


    # print(coeff.db_dt[k])
    # print("dbdt")
    # print(numpy.sum(coeff.kappa * coeff.db_dt, axis=0))
    # print("kappa * dbdt sum")

    # print(f"Cv must be : {((1/coeff.eta[0]) -1) * numpy.sum(coeff.kappa * coeff.db_dt, axis=0) * mesh.dt }")


    # print(coeff.eta[0])
    # print("eta")
    # print(coeff.chi[k])
    # print("chi")

    # print(numpy.sum(coeff.chi, axis=0))
    # print("sum chi")
    # print(f"max chi =  {numpy.max(coeff.chi)}")

    # print(coeff.chi[:, 0])
    # print("chi spectrum")

    # print(coeff.q[k])
    # print("q")


    # print(coeff.kappa[k]*coeff.beta[k] + (coeff.eta * coeff.chi[k] *-numpy.sum(coeff.kappa * coeff.beta, axis=0)))
    # print("q should be this")

    for g in range(0, mesh.ng):
        fiss[g] += tools.dbl(coeff.sig_f[g])*(prev_I[g])

        
    source[0:2*mesh.nx] = (dx*( mass_global @ coeff.S[k])
                + (tools.dbl(coeff.eta) * tools.dbl(coeff.chi[k])
                * dx* (mass_global @ numpy.sum(fiss, axis=0))))
    # add BCs
    source[0] += mesh.F_BC[k, 0]
    source[(2*mesh.nx)-1] += -mesh.F_BC[k, 1]

    # print(f"D dimensions : {coeff.D.shape}")

    # source[2*mesh.nx] += coeff.D[k, 0]*mesh.I_BC[k, 0]
    # source[-1] +=        -coeff.D[k, -1]*mesh.I_BC[k, 1]

    source[2*mesh.nx] += mesh.I_BC[k, 0]
    source[-1] +=        -mesh.I_BC[k, 1]

    # print("D IS SET TO ZERO FOR TESTING IN SOURCE")



    return source

def get_LO_source():
    # returns numpy array of size (Nx, 1)
    return 0


def high_order_assembly(mesh : tools.Discretization, coeff : tools.MG_coefficients, last_iter_I, k):
    system = tools.Global_system()
    # system.mat = assemble_global_matrix(mesh, coeff.sig_a+coeff.sig_f[k], coeff.D[k])

    # print(coeff.sig_a+coeff.sig_f[k])
    # print("sigma in HO assembly")

    system.mat = global_mat_elementwise(mesh, coeff.sig_a+coeff.sig_f[k], coeff.D[k])
    # print(f"D IS SET TO ZERO FOR TESTING")
    system.src = get_HO_source(mesh, last_iter_I, coeff, k)

    # print(system.mat.todense()[0:6,:])
    # print("Global matrix snippet")

    # print(system.src[:])
    # print("source")

    print(coeff.D[k])
    print("D")

    print(coeff.sig_a)
    print("sigma_a")
    
    print(coeff.sig_f[k])
    print("sigma_f")


    return system

def get_grey_constants():
    # TODO
    # calculate grey diffusion constants
    return 0

def low_order_assembly():
    # TODO
    # conservation = moment0_stencil(sigma, mesh)
    # diffusion    = moment1_stencil(D, mesh)

    # conservation_bc = moment0_bc(sigma, mesh)
    # diffusion_bc    = moment1_bc(D, mesh)

    # source = moment0_source(sigma, previous, mesh)

    # global_system = assemble_global(interior = [...] bc = [...] q=source)

    # set up matrix + source vector
    # pcg or other iterative solve
    return 0

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

    # print(coeff.S[0])
    # print("Reemission source")

    # print(coeff.q[0])
    # print("q source")
    
    updated_solution = numpy.zeros((mesh.ng, 4*mesh.nx))
    last_iteration = numpy.zeros((mesh.ng, 4*mesh.nx))
    last_iteration[:,:] = sol_prev.vec[:].copy()
    
    change = [1]
    iter = 0

    while (numpy.max(change) > mesh.eps) :
    # while (iter < 20) :
        iter += 1
        print(f"Iteration {iter}, change = {numpy.max(change)}")

        print("spectral radius:")
        print(coeff.eta * numpy.sum((coeff.chi * coeff.sig_f)/(coeff.sig_a + coeff.sig_f), axis=0))

        print("eta:")
        print(coeff.eta)

        print("siga")
        print(coeff.sig_a)

        for k in range(0, mesh.ng):
            sys = high_order_assembly(mesh, coeff, last_iteration[:, :2*mesh.nx], k)
            # print(I_prev[k, 0:6] / (mesh.C*mesh.dt))
            # print("S - q should be this")


            # updated_solution[k, :], b = sparse.linalg.lgmres(sys.mat, sys.src, atol=mesh.eps, rtol = mesh.eps, x0=last_iteration[k])
            # updated_solution[k,:] = sparse.linalg.spsolve(sys.mat, sys.src)
            updated_solution[k,:] = sparse.linalg.inv(sys.mat) @ sys.src
            print(f"Solved group {k} directly")
        # if b !=0:
        #     print(f"GMRES error, {b}")
        
        # print(f"Call = {iter}")

        # if iter ==1:
        #     print("Iteration 1 complete")
        #     print("Intensity group 0, cells 0-3:")
        #     print("  Left values:", updated_solution[0, 0:6:2])
        #     print("  Right values:", updated_solution[0, 1:6:2])
        #     print("Flux group 0, cells 0-3:")
        #     print("  Left values:", updated_solution[0, 2*mesh.nx:2*mesh.nx+6:2])
        #     print("  Right values:", updated_solution[0, 2*mesh.nx+1:2*mesh.nx+6:2])

        last_iteration[:] = updated_solution.copy()

        diff = abs((updated_solution / last_iteration)- 1) 
        
        # diff =  abs(updated_solution - last_iteration)
        # diff = diff / numpy.maximum(abs(last_iteration), 1e-2)
        change = numpy.linalg.norm(diff, 2, axis=1)
        print(change)
        print("change vector")
        

        print()

    
        
    return updated_solution


def accelerated_loop():
    # TODO

    # while change > tol, do:
        # high-order iteration
        # for k in [groups]:
            # generate 'fission' source from previous solution
            # high_order_assembly (constants, previous solution)
            # solve linear system
            # store I_{x+1/2}
        # generate grey constants
        # generate grey source
        # low_order_assembly (constants, previous solution)
        # solve linear system
        # use EQ spectrum to scale group error from grey error
        # compute iterate I_{x+1}
        # calculate norm(change)
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

def solve(mesh : tools.Discretization, scale : tools.Scales, opacity, acc=0):
    # TODO
    dt = mesh.dt
    nu = mesh.groups

    # gets intensity, flux, and temperature at each provided time stop 
    transport_output = []
    temp_output = tools.Transport_solution()

    if acc == 0:
        # 
        
        # unaccelerated iteration scheme

        


        pass
    else:
        pass
        # accelerated iteration scheme
    

    # scale all inputs
    # for each time step:
        # compute temperature-dependent coefficients using previous temperature
        # (un)accelerated_loop() based on specified method
        # calculate temperature change
        # populate time-dependent temperature & intensity vector at this time
    # rescale for outputs
    return 0



    
        
    
