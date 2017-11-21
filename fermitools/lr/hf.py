from ..math import einsum


def t_d1(pov):
    return -einsum('...ia->ia...', pov)


def a_d1d1_rf(foo, fvv, govov):

    def _sigma(r1):
        return (
            + einsum('ab,ib...->ia...', fvv, r1)
            - einsum('ij,ja...->ia...', foo, r1)
            - einsum('ibja,jb...->ia...', govov, r1))

    return _sigma


def b_d1d1_rf(goovv):

    def _sigma(r1):
        return einsum('ijab,jb...->ia...', goovv, r1)

    return _sigma
