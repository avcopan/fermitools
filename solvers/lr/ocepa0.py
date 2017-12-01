from .odc12 import solve_spectrum as spectrum_
from .odc12 import solve_static_response as static_response_


def solve_spectrum(nroots, nocc, norb, a11_, b11_, a12_, b12_, a21_, b21_,
                   a22_, x11_):

    return spectrum_(
            nroots=nroots, nocc=nocc, norb=norb, a11_=a11_, b11_=b11_,
            a12_=a12_, b12_=b12_, a21_=a21_, b21_=b21_, a22_=a22_, b22_=_null,
            x11_=x11_)


def solve_static_response(nocc, norb, a11_, b11_, a12_, b12_, a21_, b21_, a22_,
                          pg1, pg2):

    return static_response_(
            nocc=nocc, norb=norb, a11_=a11_, b11_=b11_, a12_=a12_, b12_=b12_,
            a21_=a21_, b21_=b21_, a22_=a22_, b22_=_null, pg1=pg1, pg2=pg2)


# Private
def _null(r):
    return 0.
