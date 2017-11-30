from ..math import einsum


def t_(pov):
    return -einsum('...ia->ia...', pov)


def a_(foo, fvv, govov):

    def _sigma(r1):
        return (
            + einsum('ab,ib...->ia...', fvv, r1)
            - einsum('ij,ja...->ia...', foo, r1)
            - einsum('ibja,jb...->ia...', govov, r1))

    return _sigma


def b_(goovv):

    def _sigma(r1):
        return einsum('ijab,jb...->ia...', goovv, r1)

    return _sigma
