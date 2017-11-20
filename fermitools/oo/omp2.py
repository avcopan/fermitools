from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm


def onebody_correlation_density(t2):
    """ the one-body correlation density matrix

    :param t2: two-body amplitudes
    :type t2: numpy.ndarray

    :returns: occupied and virtual blocks of correlation density
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    m1oo = - 1./2 * einsum('jkab,ikab->ij', t2, t2)
    m1vv = + 1./2 * einsum('ijac,ijbc->ab', t2, t2)
    return m1oo, m1vv


def twobody_amplitude_gradient(goovv, foo, fvv, t2):
    return (goovv
            + asm("2/3")(einsum('ac,ijcb->ijab', fvv, t2))
            - asm("0/1")(einsum('ki,kjab->ijab', foo, t2)))
