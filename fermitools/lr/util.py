import numpy

from ..math.sigma import subtract


def transition_dipole(s, d, pg, x, y):
    m = subtract(s, d)
    norms = numpy.diag(numpy.dot(x.T, m(x)) - numpy.dot(y.T, m(y)))
    t = numpy.dot(x.T, pg) + numpy.dot(y.T, pg)
    return norms[:, None] * t * t
