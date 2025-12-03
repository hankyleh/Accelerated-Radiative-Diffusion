from physics import C
import tools
import physics # type: ignore
import numpy
import scipy.sparse as sparse


def assemble_global_matrix(mesh  : tools.Discretization, sigma, D):
    # LHS of zeroth moment equation plus Fick's Law.
    # works in the general case-- high order or low order.

    # mesh -- variable of type "Discretization"
    # sigma-- numpy array of size (Nx, 1)
    # D    -- numpy array of size (Nx, 1)
    nx = mesh.nx
    dx = mesh.dx
    
    global_matrix = sparse.lil_array((4*nx, 4*nx))
    global_source = sparse.lil_array((4*nx, 1))

    # interior elements
    b_0i = tools.b_0i
    b_0f = tools.b_0f
    b_1i = tools.b_1i
    b_1f = tools.b_1f
    a  = tools.a_f 

    M    = (1/6)*numpy.array([[2, 1],
                              [1, 2]])
    M_wide = (1/6)*numpy.array([[0, 2, 1, 0],
                              [0, 1, 2, 0]])

    local_operator = numpy.zeros((4, 4))
    intensity_op = numpy.zeros((4, 4))

    # interior elements
    for i in range(1, nx-1):
        # i = index of current interior cell
        l_nbr = (2*i)-1
        r_nbr = (2*i)+2
        shft  = (2*nx)

        # zeroth moment, intensity
        global_matrix[2*i:2*i + 2, l_nbr:r_nbr+1] += b_0i + (sigma[i]*dx*M_wide)

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
        prev_soln, 
        coeff : tools.MG_coefficients, 
        k     : int):
    source = numpy.zeros((4*mesh.nx))

    # compute fission source, add to 'S'
    source[0:2*mesh.nx] = (coeff.S[k] 
                + (tools.dbl(coeff.eta[k]) * tools.dbl(coeff.chi[k])
                * numpy.sum(tools.dbl(coeff.sig_f) * prev_soln[:, 0:2*mesh.nx], axis=0)))
    # print(numpy.sum(tools.dbl(coeff.sig_f) * prev_soln[:, 0:2*mesh.nx], axis=0))
    # add BCs
    source[0] += mesh.F_BC[k, 0]
    source[(2*mesh.nx)-1] += - mesh.F_BC[k, 1]

    source[2*mesh.nx] += coeff.D[k, 0]*mesh.I_BC[k, 0]
    source[-1] +=        -coeff.D[k, -1]*mesh.I_BC[k, 1]
    return source

def get_LO_source():
    # returns numpy array of size (Nx, 1)
    return 0


def high_order_assembly(mesh : tools.Discretization, coeff : tools.MG_coefficients, last_iter_I, k):
    system = tools.Global_system()
    system.mat = assemble_global_matrix(mesh, coeff.sig_a+coeff.sig_f[k], coeff.D[k])
    system.src = get_HO_source(mesh, last_iter_I, coeff, k)
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

def unaccelerated_loop(mesh : tools.Discretization, sol_prev, T_prev, kappa, Cv, Q):
    coeff = tools.MG_coefficients(mesh)
    I_prev = sol_prev[:, :2*mesh.nx]
    coeff.assign(mesh, kappa, I_prev, T_prev, Cv, Q)


    last_iteration = sol_prev.copy()
    updated_solution = sol_prev.copy()

    # print(last_iteration)
    
    # print(updated_solution.shape)

    change = [1]
    iter = 0

    while (numpy.max(change) > mesh.eps) :
        iter += 1
        print(f"Iteration {iter}, change = {numpy.max(change)}")

        for k in range(0, mesh.ng):
            sys = high_order_assembly(mesh, coeff, last_iteration[:, :2*mesh.nx], k)
            updated_solution[k, :], b = sparse.linalg.gmres(sys.mat, sys.src, rtol=mesh.eps, x0=last_iteration[k])

        diff =  abs(updated_solution - last_iteration)
        diff = diff/ (1*(last_iteration==0) + (last_iteration!=0)*last_iteration)
        change = numpy.linalg.norm(diff, 2, axis=0)
        last_iteration = updated_solution.copy()
        
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

def solve(mesh : tools.Discretization, scale : tools.Scales, opacity, acc=0):
    # TODO
    dt = mesh.dt
    nu = mesh.groups

    # gets intensity, flux, and temperature at each provided time stop
    transport_output = numpy.zeros((numpy.sum(mesh.nt), mesh.ng, 4*mesh.nx))
    temp_output = numpy.zeros((numpy.sum(mesh.nt), mesh.nx))

    if acc == 0:

        
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



    
        
    
