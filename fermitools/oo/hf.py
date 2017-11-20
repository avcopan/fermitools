import numpy


def electronic_energy(hoo, goooo):
    return (+ numpy.trace(hoo)
            + 1./2 * numpy.trace(numpy.trace(goooo, axis1=0, axis2=2)))


def fock_oo(hoo, goooo):
    return hoo + numpy.trace(goooo, axis1=0, axis2=2)


def fock_vv(hvv, govov):
    return hvv + numpy.trace(govov, axis1=0, axis2=2)


def fock_ov(hov, gooov):
    return hov + numpy.trace(gooov, axis1=0, axis2=2)


def orbital_gradient(hov, gooov):
    return fock_ov(hov, gooov)
