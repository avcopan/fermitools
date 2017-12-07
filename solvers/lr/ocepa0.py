from .odc12 import solve_spectrum as spectrum_
from .odc12 import solve_static_response as static_response_


def solve_spectrum(nroots, nocc, norb, a11, b11, a12, b12, a21, b21, a22, x11):

    return spectrum_(
            nroots=nroots, nocc=nocc, norb=norb, a11=a11, b11=b11, a12=a12,
            b12=b12, a21=a21, b21=b21, a22=a22, b22=_null, x11=x11)


def solve_static_response(nocc, norb, a11, b11, a12, b12, a21, b21, a22, pg1,
                          pg2):

    return static_response_(
            nocc=nocc, norb=norb, a11=a11, b11=b11, a12=a12, b12=b12, a21=a21,
            b21=b21, a22=a22, b22=_null, pg1=pg1, pg2=pg2)


# Private
def _null(r):
    return 0.
