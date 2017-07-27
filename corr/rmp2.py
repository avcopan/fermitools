import numpy


def correlation_energy(n, g, e):
    """second-order Møller-Plesset correlation energy

    :param n: number of electrons
    :type n: int
    :param g: MO-basis electron repulsion integrals, in bra-ket notation
    :type g: numpy.ndarray
    :param e: orbital energies
    :type e: numpy.ndarray

    :rtype: float
    """
    o = slice(None, n)
    v = slice(n, None)
    x = numpy.newaxis

    t = (g[o, o, v, v]
         / (e[o, x, x, x] + e[x, o, x, x] - e[x, x, v, x] - e[x, x, x, v]))

    u = 2. * t - numpy.transpose(t, (0, 1, 3, 2))

    return numpy.sum(g[o, o, v, v] * u)
