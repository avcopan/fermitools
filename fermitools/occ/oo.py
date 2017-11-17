import numpy


def fock(h, g, m1):
    return h + numpy.tensordot(g, m1, axes=((1, 3), (0, 1)))


def onebody_determinant_density(norb, nocc):
    m1 = numpy.zeros((norb, norb))
    m1[:nocc, :nocc] = numpy.eye(nocc)
    return m1
