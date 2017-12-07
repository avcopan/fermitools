from ..math import einsum
from ..math import broadcast_sum


def pg(pov):
    return -einsum('...ia->ia...', pov)


def a_sigma(foo, fvv, govov):

    def _a(r):
        return (
            + einsum('ab,ib...->ia...', fvv, r)
            - einsum('ij,ja...->ia...', foo, r)
            - einsum('ibja,jb...->ia...', govov, r))

    return _a


def b_sigma(goovv):

    def _b(r):
        return einsum('ijab,jb...->ia...', goovv, r)

    return _b


def pc_sigma(eo, ev):

    def _pc(w):

        def __pc(r):
            return r / broadcast_sum({0: -eo, 1: +ev, 2: -w})

        return __pc

    return _pc
