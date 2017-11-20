from .omp2 import onebody_correlation_density
from ..math import einsum


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


__all__ = ['onebody_correlation_density']
