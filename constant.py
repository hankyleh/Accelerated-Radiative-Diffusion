import numpy

H = numpy.float128(4.135667696e-15) # eV * s
K = numpy.float128(8.617333262e-5)  # eV / K
C = numpy.float128(2.99792458e10)   # cm / s
SIG_R = numpy.float128(3.53916934e7) # eV / (cm^2 * s * K^4)
A_R = numpy.float128(4.72215928e-3)  # eV / (cm^2 * s * K^4)

class discretization:
    def __init__(self):
        self.groups = numpy.array([0.000, 1e10])*(1000/H)
        self.ng = 1
        self.x_length = 4
        self.t_stops = numpy.array([1e-2, 2e-2])
        self.dx = 0.4 # cm
        self.dt = 2e-3 # shakes
        self.nx = 10
        self.nt = numpy.array([50, 100])

        self.cell_centers = numpy.linspace(0.2, 3.8, 10)
        self.cell_edges   = numpy.linspace(0, 4, 11)

        # TODO -- change all these to manual definition as in below. Right now recursive definition.

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name=="groups":
            self.__dict__["ng"] = len(value) - 1
        if name=="dt":
            self.__dict__["nt"] = self.__dict__["t_stops"]/value
        if name=="dx":
            self.__dict__["nx"] = self.__dict__["t_stops"]/value




class scales:
    t = 1
    I = 1
    B = 1
    E = 1

class mg_coefficients:
    def __init__(self, mesh):
        eta = 1 
        chi = numpy.zeros(len(mesh.groups) - 1)
        q   = numpy.zeros(len(mesh.groups) - 1)