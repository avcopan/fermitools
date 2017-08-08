import numpy


def doubles_correlation_energy(goovv, t2):
    return numpy.vdot(goovv, t2) / 4.
