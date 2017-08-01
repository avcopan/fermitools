import numpy


def correlation_energy(na, nb, aag, abg, bbg, ae, be):
    """second-order Møller-Plesset correlation energy

    :param na: number of alpha electrons
    :type na: int
    :param nb: number of beta electrons
    :type nb: int
    :param aag: <a,a|a,a> electron repulsion integrals
    :type aag: numpy.ndarray
    :param abg: <a,b|a,b> electron repulsion integrals
    :type abg: numpy.ndarray
    :param bbg: <b,b|b,b> electron repulsion integrals
    :type bbg: numpy.ndarray
    :param ae: alpha orbital energies
    :type ae: numpy.ndarray
    :param be: beta orbital energies
    :type be: numpy.ndarray

    :rtype: float
    """
    ao = slice(None, na)
    bo = slice(None, nb)
    av = slice(na, None)
    bv = slice(nb, None)
    x = numpy.newaxis

    aat = (aag[ao, ao, av, av] /
           (+ ae[ao, x, x, x] + ae[x, ao, x, x]
            - ae[x, x, av, x] - ae[x, x, x, av]))

    abt = (abg[ao, bo, av, bv] /
           (+ ae[ao, x, x, x] + be[x, bo, x, x]
            - ae[x, x, av, x] - be[x, x, x, bv]))

    bbt = (bbg[bo, bo, bv, bv] /
           (+ be[bo, x, x, x] + be[x, bo, x, x]
            - be[x, x, bv, x] - be[x, x, x, bv]))

    aau = aat - numpy.transpose(aat, (0, 1, 3, 2))
    bbu = bbt - numpy.transpose(bbt, (0, 1, 3, 2))

    aaenergy = numpy.sum(aag[ao, ao, av, av] * aau) / 2.
    abenergy = numpy.sum(abg[ao, bo, av, bv] * abt)
    bbenergy = numpy.sum(bbg[bo, bo, bv, bv] * bbu) / 2.

    return aaenergy + abenergy + bbenergy
