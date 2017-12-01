from .odc12 import solve_spectrum as spectrum_
from .odc12 import solve_static_response as static_response_


def solve_spectrum(nroots, nocc, norb, a11_, b11_, a_d1d2_, b_d1d2_, a_d2d1_,
                   b_d2d1_, a_d2d2_, x1_):

    return spectrum_(
            nroots=nroots, nocc=nocc, norb=norb, a11_=a11_, b11_=b11_,
            a_d1d2_=a_d1d2_, b_d1d2_=b_d1d2_, a_d2d1_=a_d2d1_, b_d2d1_=b_d2d1_,
            a_d2d2_=a_d2d2_, b_d2d2_=_null, x1_=x1_)


def solve_static_response(nocc, norb, a11_, b11_, a_d1d2_, b_d1d2_, a_d2d1_,
                          b_d2d1_, a_d2d2_, pg1, pg2):

    return static_response_(
            nocc=nocc, norb=norb, a11_=a11_, b11_=b11_, a_d1d2_=a_d1d2_,
            b_d1d2_=b_d1d2_, a_d2d1_=a_d2d1_, b_d2d1_=b_d2d1_, a_d2d2_=a_d2d2_,
            b_d2d2_=_null, pg1=pg1, pg2=pg2)


# Private
def _null(r):
    return 0.
