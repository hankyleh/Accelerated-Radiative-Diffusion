from physics import C
import tools
import physics # type: ignore
import numpy
import scipy.sparse as sparse
import scipy
import copy
from matplotlib import pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def assemble_global_matrix(mesh  : tools.Discretization, sigma, D):
    # LHS of zeroth moment equation plus Fick's Law.
    # works in the general case-- high order or low order.

    nx = mesh.nx
    dx = mesh.dx
    
    global_matrix = sparse.dok_array((4*nx, 4*nx))
    # interior elements

    B_1 = tools.B_1
    B_2 = tools.B_2

    M_wide = tools.M_wide

    # interior elements
    for i in range(1, nx-1):
        R_prev = (2*i)-1
        L_next = (2*i)+2
        shift  = (2*nx)

        # zeroth moment, intensity
        global_matrix[2*i:(2*i + 2), 
                R_prev:L_next+1] += B_1 + (sigma[i]*dx*M_wide)
        # zeroth moment, flux
        global_matrix[2*i:2*i + 2, 
                        R_prev+shift:L_next+shift+1] += B_2

        # first moment, intensity
        global_matrix[2*i +shift:2*i + 2+shift, 
                      R_prev:L_next+1] += D[i] * (B_2) 
        # first moment, flux
        global_matrix[2*i+shift:2*i+2+shift,
                      R_prev+shift:L_next+1+shift]+= (dx*M_wide 
                                                                 + D[i]*3*B_1)

    # boundary elements
    # -----------------
    # Left boundary
    # zeroth, intensity
    global_matrix[0:2, 0:3] += B_1[:, 1:] + (sigma[0]*dx*M_wide[:, 1:])
    # zeroth, flux
    global_matrix[0:2, shift:3+shift] +=  B_2[:, 1:]
    # first, intensity
    global_matrix[shift:2+shift, 
                  0:3] += D[0] * (B_2[:, 1:]) 
    # first, flux
    global_matrix[shift:2+shift,
                  shift:3+shift] += ((dx*M_wide[:, 1:]) +
                                               (D[0]*3*B_1[:, 1:]))
    # Right boundary
    # zeroth, intensity
    global_matrix[shift-2:shift, 
                  shift-3:shift] += (B_1[:, 0:-1] + 
                                     (sigma[-1]*dx*M_wide[:, 0:-1]))
    # zeroth, flux
    global_matrix[shift-2:shift, 
                  -3:] +=  B_2[:, 0:-1] 
    # first, intensity
    global_matrix[-2:, 
                  shift-3:shift] += D[-1] * (B_2[:, 0:-1] ) 
    # first, flux
    global_matrix[-2:, 
                  -3:] += ((dx*M_wide[:, 0:-1]) + 
                                        (D[-1]*3*B_1[:, 0:-1]))
    return global_matrix.tocsr()

def get_HO_source(
        mesh  : tools.Discretization, 
        prev_I, 
        coeff : tools.MG_coefficients, 
        k     : int):
    source = numpy.zeros((4*mesh.nx))
    dx = mesh.dx
    nx = mesh.nx

    # maybe compute mass_global outside loop TODO
    M = tools.M

    mass_global = sparse.lil_array((2*nx, 2*nx))
    for i in range(0, nx):
        mass_global[2*i : 2*i+2, 2*i : 2*i+2] += M[:]
    mass_global[:] = mass_global.tocsr()

    # total fission
    fiss = numpy.zeros((mesh.ng, 2*nx))
    for g in range(0, mesh.ng):
        fiss[g] += tools.dbl(coeff.sig_f[g])*(prev_I[g])

    # source = q_k + I^n/cdt + fiss
    source[0:2*mesh.nx] += (dx*( mass_global @ coeff.S[k])
                + (tools.dbl(coeff.eta) * tools.dbl(coeff.chi[k])
                * dx* (mass_global @ numpy.sum(fiss, axis=0))))

    # add BCs
    source[0]             += mesh.F_BC[k, 0]
    source[(2*mesh.nx)-1] += -mesh.F_BC[k, 1]


    source[2*mesh.nx] += coeff.D[k, 0] * mesh.I_BC[k, 0]
    source[-1] +=      - coeff.D[k, -1] * mesh.I_BC[k, 1]

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
    system.mat = assemble_global_matrix(mesh, coeff.sig_a+coeff.sig_f[k], coeff.D[k])
    system.src = get_HO_source(mesh, last_iter_I, coeff, k)
    return system


def assemble_LO(mesh : tools.Discretization, coeff : tools.Grey_coeff):
    system = tools.Global_system()
    system.mat = assemble_global_matrix(mesh, coeff.sig_a + (1-coeff.eta)*coeff.sigf_avg, coeff.D_avg)
    system.src = get_LO_source(mesh, coeff)
    return system

def unaccelerated_loop(mesh : tools.Discretization, 
                       sol_prev : tools.Transport_solution, 
                       coeff : tools.MG_coefficients, 
                       flags : dict,
                       guess  = None):

    if guess is None:
        last_iteration = copy.deepcopy(sol_prev)
    else:
        last_iteration = guess
    updated_solution =  copy.deepcopy(sol_prev)
    
    change = [1]

    if flags["coarse"] is True:
        gmtol = mesh.eps
        tol = mesh.eps_c
    else:
        gmtol = mesh.eps_f
        tol   = mesh.eps

    iter = 0
    while (numpy.max(change) > tol) :
        iter += 1
        for k in range(0, mesh.ng):
            sys = assemble_HO(mesh, coeff, last_iteration.intensity, k)
            if flags["mat_method"] == "lu":
                updated_solution.vec[k, :] = sparse.linalg.spsolve(sys.mat, sys.src)
            elif flags["mat_method"] == "gmres":
                updated_solution.vec[k, :], b = sparse.linalg.gmres(sys.mat, sys.src, x0=last_iteration.vec[k], rtol = gmtol)
            elif flags["mat_method"] == "inv":
                updated_solution.vec[k, :] = numpy.matmul(scipy.linalg.inv(sys.mat.todense()), sys.src)
            else:
                raise ValueError("Invalid solution method provided")

        diff = abs((updated_solution.intensity / last_iteration.intensity)- 1)
        diff = numpy.nan_to_num(diff)

        # norm along space (keep all groups)
        change = scipy.linalg.norm(diff, 2, axis=1)                       
        # norm along groups
        change = numpy.append(change, scipy.linalg.norm(diff, 2, axis=0)) 

        if flags["print_iter_change"] is True:
            print(change)
            print("iteration change")

        last_iteration = copy.deepcopy(updated_solution)

        def print_update():
            print(f"Step {flags["time_frac"]:<8}"+
                  f"Iteration {iter:<6}"+
                  f"Max. rel. change {numpy.max(change):.4e}  "+
                  f"{"MANUALLY ASSIGNED OPACITY" * (flags["manual_kappa"] is not False)}"+
                  f"'{flags["mat_method"]}' LD solution method  "+
                  f"{"Using Accelerated Algorithm"*(flags["accelerated"] is not False)}",
                        end="\r", flush=True)

        if iter % flags["printing_interval"] == 0:
            print_update()
    # prints again when converged
    print_update()
    return updated_solution.vec, iter


def accelerated_loop(mesh : tools.Discretization, 
                       sol_prev : tools.Transport_solution, 
                       coeff : tools.MG_coefficients, 
                       flags : dict,
                       guess  = None):

    I_prev = sol_prev.intensity[:]
    if guess is None:
        last_iteration = copy.deepcopy(sol_prev)
    else:
        last_iteration = guess
    updated_solution =  copy.deepcopy(sol_prev)
    error_soln = copy.deepcopy(sol_prev)

    grey_constants = tools.Grey_coeff(mesh)
    
    change = [1]

    if flags["coarse"] is True:
        gmtol = mesh.eps
        tol = mesh.eps_c
    else:
        gmtol = mesh.eps_f
        tol   = mesh.eps

    iter = 0
    while (numpy.max(change) > tol):
        iter += 1

        for k in range(0, mesh.ng):
            sys = assemble_HO(mesh, coeff, last_iteration.intensity, k)
            if flags["mat_method"] == "lu":
                updated_solution.vec[k, :] = sparse.linalg.spsolve(sys.mat, sys.src)
            elif flags["mat_method"] == "gmres":
                updated_solution.vec[k, :], b = sparse.linalg.gmres(sys.mat, sys.src, x0=last_iteration.vec[k], rtol = gmtol)
            elif flags["mat_method"] == "inv":
                updated_solution.vec[k, :] = numpy.matmul(scipy.linalg.inv(sys.mat.todense()), sys.src)
            else:
                raise ValueError("Invalid solution method provided")
        
        grey_constants.assign(mesh, coeff, updated_solution, last_iteration)
        sys_grey = assemble_LO(mesh, grey_constants)

        if flags["mat_method"] == "lu":
            error_soln.vec[:] = sparse.linalg.spsolve(sys_grey.mat, sys_grey.src)
        elif flags["mat_method"] == "gmres":
            error_soln.vec[:], b = sparse.linalg.gmres(sys_grey.mat, sys_grey.src, x0=last_iteration.vec[k], rtol = gmtol)
        elif flags["mat_method"] == "inv":
            error_soln.vec[:] = numpy.matmul(scipy.linalg.inv(sys_grey.mat.todense()), sys_grey.src)
        else:
            raise ValueError("Invalid solution method provided")
        
        updated_solution.vec[:] += error_soln.vec[:]*numpy.tile(tools.dbl(grey_constants.spectrum), reps = (1, 2))
    

        diff = abs((updated_solution.intensity / last_iteration.intensity)- 1)
        diff = numpy.nan_to_num(diff)
        
        change = scipy.linalg.norm(diff, 2, axis=1)
        change = numpy.append(change, scipy.linalg.norm(diff, 2, axis=0))
        last_iteration = copy.deepcopy(updated_solution)

        def print_update():
            print(f"\rStep {flags["time_frac"]:<8}"+
                  f"Iteration {iter:<6}"+
                  f"Max. rel. change {numpy.max(change):.4e}  "+
                  f"{"MANUALLY ASSIGNED OPACITY" * (flags["manual_kappa"] is not False)}"+
                  f"'{flags["mat_method"]}' LD solution method  "+
                  f"{"Using Accelerated Algorithm"*(flags["accelerated"] is not False)}", 
                  end = "", flush=True)

        if iter % flags["printing_interval"] == 0:
            print_update()
    # prints again when converged
    print_update()
    return updated_solution.vec, iter


def loop(mesh : tools.Discretization, 
         sol_prev : tools.Transport_solution, 
         coeff : tools.MG_coefficients, 
         flags : dict,
         guess = None):

    if flags["accelerated"] == True:
        transport_sol, inners  = accelerated_loop(mesh, sol_prev, coeff, flags, guess)
    else:
        transport_sol, inners  = unaccelerated_loop(mesh, sol_prev, coeff, flags, guess)

    return transport_sol, inners
    

def update_temperature(mesh : tools.Discretization, 
                       coeff : tools.MG_coefficients, 
                       soln : tools.Transport_solution,
                       Cv,
                       Q = 0):
    
    temp_change = (
        numpy.sum(coeff.kappa * (soln.cell_center_i - coeff.beta), axis=0) + Q
    )/(
        (Cv/mesh.dt) + numpy.sum(coeff.kappa * coeff.db_dt, axis=0)
    )

    # if (sum(math.isnan(temp_change[j]) for j in range(0, mesh.nx)) > 0):
    #     print(temp_change[:])
    #     print("temp change")
    #     print(coeff.kappa[:, 0:3])
    #     print("kappa sample")
    #     # print(soln.cell_center_i[0:6, 0:6])
    #     # print("cell center i sample")
    #     # print(coeff.beta[0:6, 0:6])
    #     # print("beta sample")
    #     print((Cv/mesh.dt) + numpy.sum(coeff.kappa * coeff.db_dt, axis=0))
    #     print("denominator")
    #     print( numpy.sum(coeff.kappa * (soln.cell_center_i - coeff.beta), axis=0) + Q)
    #     print("numerator")
    #     print(numpy.sum(coeff.kappa * coeff.db_dt, axis=0))
    #     print("sum term")
    #     print(coeff.db_dt[:, 0:3])
    #     print("dbdt")

    #     raise ValueError("NaN dT encountered")
    

    return temp_change

def solve_diffusion(mesh : tools.Discretization, 
                        scale : tools.Scales, opacity, 
                        IC : tools.Transport_solution, 
                        T_init, 
                        Cv, 
                        Q=0, 
                        accelerated=False, 
                        first_step_only=False,
                        exact_inverse=False,
                        manual_kappa = False,
                        print_T_change = False,
                        print_kappa = False, 
                        mat_method = "lu",
                        printing_interval = 5,
                        print_coeff = False,
                        print_iter_change = False,
                        coarse = False,
                        Newton = True):
    print("Beginning unaccelerated iteration")

    flags = {
        "accelerated"    : accelerated    ,
        "first_step_only": first_step_only,
        "manual_kappa"   : manual_kappa   ,
        "print_T_change" : print_T_change ,
        "print_kappa"    : print_kappa    ,
        "mat_method"     : mat_method,
        "printing_interval" : printing_interval,
        "print_coeff"    : print_coeff,
        "print_iter_change" : print_iter_change,
        "coarse" : coarse
    }

    dt = mesh.dt
    nu = mesh.groups

    nt = mesh.nt
    k_star = 27 # make this an input TODO

    transport_output = []
    temp_output = []
    iters_log = numpy.zeros((2, numpy.sum(mesh.nt))).astype(int)
    transport_iters = copy.deepcopy(IC)
    sol_prev = copy.deepcopy(IC)

    T_n = copy.deepcopy(T_init)

    kappa = numpy.zeros((mesh.ng, mesh.nx))
    dT = numpy.zeros(mesh.nx)
    beta = copy.deepcopy(dT)

    for stop in range(0, (first_step_only == False)*len(nt) 
                        + (first_step_only == True)):

        print(f"Starting steps towards {mesh.t_stops[stop]}")

        for step in range(0, nt[stop]):
            inners = 0
            flags["time_frac"] = f"{(step + numpy.sum(mesh.nt[0:stop])) + 1}/{numpy.sum(mesh.nt)}"

            if manual_kappa is False:
                kappa[:] = opacity(mesh, T_n, k_star)
            else:
                    kappa[:] = manual_kappa
            if print_kappa is True:
                    print(kappa[:, 0])
                    print("Group opacity in first cell")

            coeff = tools.MG_coefficients(mesh)
            coeff.assign(mesh, kappa, sol_prev, T_n, Cv, Q)

            if print_coeff is True:
                print()
                print(coeff.chi[:, 0])
                print(coeff.chi[:, -1])
                print("Chi_k, first cell; last cell")
                print(coeff.eta[0])
                print(coeff.eta[-1])
                print("Eta, first cell; last cell")
                print(coeff.sig_a)
                print("Absorption sigma")
                print(coeff.sig_f[:, 0])
                print(coeff.sig_f[:, -1])
                print("sig_f, first cell; last cell")


            def check_nan(mesh : tools.Discretization, soln : tools.Transport_solution, prev = None):
                if (sum(math.isnan(soln.intensity[i, j]) for i in range (0, mesh.ng) for j in range(0, 2*mesh.nx)) > 0):
                    print(soln.intensity)
                    print("Cell-edge flux")
                    print(soln.vec[0:6, 0:6])
                    print("Sample of transport solution")
                    print(soln.cell_center_i)
                    print("Cell-centered flux")
                    print(kappa[:, 0:6])
                    print("opacity sample")
                    print(T_n[:])
                    print("Temperature")
                    print(dT[:])
                    print("most recent temperature change")

                    plt.figure()
                    ax = plt.gca()
                    lines = tools.LD_plottable(mesh, (1/mesh.C)*physics.ev_to_erg*soln.vec)
                    tools.plot_LD_groups(ax, mesh, lines.intensity, range(0, mesh.ng))
                    plt.title(f"Energy density")
                    plt.xlabel("x [cm]")
                    plt.autoscale()


                    plt.figure()
                    plt.plot(mesh.cell_centers, mesh.K*T_n)
                    plt.xlabel("x [cm]")
                    plt.ylabel("T [eV]")
                    plt.title("Temperature")

                    if prev is not None:
                        print(prev.vec[0:6, 0:6])
                        print("previous solution sample")
                        print()
                        plt.figure()
                        ax = plt.gca()
                        lines = tools.LD_plottable(mesh, (1/mesh.C)*physics.ev_to_erg*prev.vec)
                        tools.plot_LD_groups(ax, mesh, lines.intensity, range(0, mesh.ng))
                        plt.title(f"Previous iteration Energy density")
                        plt.xlabel("x [cm]")
                        plt.autoscale()

                    plt.show()
                    raise ValueError("NaN intensity encountered")
                return 0

            if Newton is True:
                alpha = 0.05
                dbdt = coeff.db_dt
                slope = numpy.zeros((mesh.ng, mesh.nx))

                slope_change = 1
                slope_iters = 0
                # iterate until slope is good

                # check dT first
                transport_iters.vec[:,:], new_inners = loop(mesh, sol_prev, coeff, flags, guess = transport_iters)
                inners += new_inners
                dT[:] = update_temperature(mesh, coeff, transport_iters, Cv)
                if numpy.linalg.norm(dT/T_n, 2) > alpha:
                    while (slope_change > mesh.eps_c):
                        slope_iters += 1
                        flags["coarse"] = True
                        temporary_transport = copy.deepcopy(transport_iters)
                        transport_iters.vec[:,:], new_inners = loop(mesh, sol_prev, coeff, 
                                                                    flags, guess = transport_iters)
                        check_nan(mesh, transport_iters, prev=temporary_transport)
                        inners += new_inners
                        dT[:] = update_temperature(mesh, coeff, transport_iters, Cv)
                        dBeta = physics.group_planck(mesh, (T_n + dT)) - coeff.beta
                        slope = dBeta / dT
                        change_vec = numpy.sum(numpy.abs( (slope / coeff.db_dt)-1 ),axis=0)
                        change_vec = numpy.nan_to_num(change_vec)
                        slope_change = numpy.max(change_vec)
                        if slope_iters > 40:
                            # TODO : temporary fix for no convergence.
                            # FIND OUT WHY THIS IS OCCURING
                            print("No Newton convergence")
                            coeff.db_dt[:, :] = dbdt
                            break

                        print()
                        print(slope_change)
                        print("slope change")
                        coeff.db_dt[:, :] = slope
                else:
                    pass
                    # dT was small; use dB/dT|_n
                # finish iterations using new slope
                flags["coarse"] = coarse


            # calculate flux and intensity, record iterations
            temporary_transport = copy.deepcopy(transport_iters)
            transport_iters.vec[:,:], new_inners = loop(mesh, sol_prev, coeff, 
                                                        flags, guess = transport_iters)
            check_nan(mesh, transport_iters, prev = temporary_transport)
            inners += new_inners
            sol_prev.vec[:] = transport_iters.vec[:].copy()

            
            
            if accelerated == True:
                iters_log[:,step + numpy.sum(mesh.nt[0:stop])] = inners * numpy.array([1, mesh.ng + 1]).transpose()
            else:
                iters_log[:,step + numpy.sum(mesh.nt[0:stop])] = inners * numpy.array([1, mesh.ng]).transpose()

                
            dT[:] = update_temperature(mesh, coeff, transport_iters, Cv)
            if print_T_change == True:
                print(dT)
                print("Temperature delta")
            T_n += copy.deepcopy(dT[:])
            coeff.assign(mesh, kappa, transport_iters, T_n, Cv, Q)
            print("")
        transport_output.append(copy.deepcopy(transport_iters))
        temp_output.append(copy.deepcopy(T_n[:]))
    
    return temp_output, transport_output, iters_log
