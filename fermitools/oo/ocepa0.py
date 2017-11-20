from .omp2 import onebody_correlation_density
from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm


def twobody_amplitude_gradient(goooo, goovv, govov, gvvvv, foo, fvv, t2):
    return (goovv
            + asm("2/3")(einsum('ac,ijcb->ijab', fvv, t2))
            - asm("0/1")(einsum('ki,kjab->ijab', foo, t2))
            + 1. / 2 * einsum("abcd,ijcd->ijab", gvvvv, t2)
            + 1. / 2 * einsum("klij,klab->ijab", goooo, t2)
            - asm("0/1|2/3")(einsum("kaic,jkbc->ijab", govov, t2)))


def twobody_cumulant(t2, blocks=('o,o,o,o', 'o,o,v,v', 'o,v,o,v', 'v,v,v,v')):
    k2 = {}
    if 'o,o,o,o' in blocks:
        k2['o,o,o,o'] = 1./2 * einsum('ijcd,klcd->ijkl', t2, t2)
    if 'o,o,v,v' in blocks:
        k2['o,o,v,v'] = t2
    if 'o,v,o,v' in blocks:
        k2['o,v,o,v'] = einsum('ikac,jkbc->jabi', t2, t2)
    if 'v,v,v,v' in blocks:
        k2['v,v,v,v'] = 1./2 * einsum('klab,klcd->abcd', t2, t2)
    return k2


__all__ = ['onebody_correlation_density', 'twobody_amplitude_gradient',
           'twobody_cumulant']
