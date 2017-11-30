import numpy


def e_(n1, e_d1d1_, e_d1d2_, e_d2d1_, e_d2d2_):

    def _sigma(r):
        r1, r2 = numpy.split(r, (n1,))
        return numpy.concatenate((e_d1d1_(r1) + e_d1d2_(r2),
                                  e_d2d1_(r1) + e_d2d2_(r2)), axis=0)

    return _sigma


def x_(n1, x_d1d1_):

    def _sigma(r12):
        r1, r2 = numpy.split(r12, (n1,))
        return numpy.concatenate((x_d1d1_(r1), r2), axis=0)

    return _sigma


def e_eff_(e_sum_, e_dif_, x_):

    def _sigma(r):
        return x_(e_sum_(x_(e_dif_(r))))

    return _sigma
