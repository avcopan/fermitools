"""spin-orbital linearized coupled-cluster doubles"""
import numpy
import toolz.functoolz as ftz

from ..math import antisymmetrizer


asym_01_23 = ftz.compose(antisymmetrizer((0, 1)), antisymmetrizer((2, 3)))


def doubles_amplitudes_update(goooo, goovv, govov, gvvvv, e2, t2):

    r2 = (- numpy.multiply(t2, e2) + numpy.array(goovv)
          + 1. / 2 * numpy.einsum("abcd,ijcd->ijab", gvvvv, t2)
          + 1. / 2 * numpy.einsum("klij,klab->ijab", goooo, t2)
          - asym_01_23(numpy.einsum("kaic,jkbc->ijab", govov, t2)))

    return t2 + r2 / e2, r2
