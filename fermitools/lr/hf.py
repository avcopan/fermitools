from ..math import einsum


def pg(pov):
    return -einsum('...ia->ia...', pov)


def a_sigma(foo, fvv, govov):

    def _a(r1):
        return (
            + einsum('ab,ib...->ia...', fvv, r1)
            - einsum('ij,ja...->ia...', foo, r1)
            - einsum('ibja,jb...->ia...', govov, r1))

    return _a


def b_sigma(goovv):

    def _b(r1):
        return einsum('ijab,jb...->ia...', goovv, r1)

    return _b
