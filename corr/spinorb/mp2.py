import numpy


def correlation_energy(n, g, e):
    """second-order Møller-Plesset correlation energy

    :param n: number of electrons
    :type n: int
    :param g: antisymmetrized (!) spin-MO electron repulsion integrals
    :type g: numpy.ndarray
    :param e: spin-orbital energies
    :type e: numpy.ndarray

    :rtype: float
    """
    o = slice(None, n)
    v = slice(n, None)
    x = numpy.newaxis

    t = (g[o, o, v, v]
         / (e[o, x, x, x] + e[x, o, x, x] - e[x, x, v, x] - e[x, x, x, v]))

    return 1. / 4 * numpy.sum(g[o, o, v, v] * t)
