import numpy


def doubles_correlation_energy(goovv, t2):
    u2 = numpy.multiply(2., t2) - numpy.transpose(t2, (0, 1, 3, 2))
    return numpy.vdot(goovv, u2)
