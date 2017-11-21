import numpy


def e_rf(n1, e_d1d1_rf, e_d1d2_rf, e_d1d2_lf, e_d2d2_rf):

    def _sigma(r):
        r1, r2 = numpy.split(r, (n1,))
        return numpy.concatenate((e_d1d1_rf(r1) + e_d1d2_rf(r2),
                                  e_d1d2_lf(r1) + e_d2d2_rf(r2)), axis=0)

    return _sigma


def x_rf(n1, x_d1d1_rf):

    def _sigma(r12):
        r1, r2 = numpy.split(r12, (n1,))
        return numpy.concatenate((x_d1d1_rf(r1), r2), axis=0)

    return _sigma


def e_eff_rf(e_sum_rf, e_dif_rf, x_rf):

    def _sigma(r):
        return x_rf(e_sum_rf(x_rf(e_dif_rf(r))))

    return _sigma
