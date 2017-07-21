import numpy


def coulomb(g, ad, bd):
    return numpy.tensordot(g, ad + bd, axes=((1, 3), (1, 0)))


def exchange(g, d):
    return numpy.tensordot(g, d, axes=((1, 2), (1, 0)))
