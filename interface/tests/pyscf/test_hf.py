import simplehf.interface.pyscf as interface

import numpy
from numpy.testing import assert_almost_equal

N = 7
BASIS = 'STO-3G'
CHARGE = +1
SPIN = 1
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

C_NORM = 2.93356458632


def test__unrestricted_orbitals():
    ac, bc = interface.hf.unrestricted_orbitals(basis=BASIS, labels=LABELS,
                                                coords=COORDS, charge=CHARGE,
                                                spin=SPIN)
    ac_norm = numpy.linalg.norm(ac)
    bc_norm = numpy.linalg.norm(bc)
    assert ac.shape == (N, N)
    assert bc.shape == (N, N)
    assert_almost_equal(ac_norm, C_NORM, decimal=10)
    assert_almost_equal(bc_norm, C_NORM, decimal=10)


def test__restricted_orbitals():
    c = interface.hf.restricted_orbitals(basis=BASIS, labels=LABELS,
                                         coords=COORDS, charge=CHARGE,
                                         spin=SPIN)
    c_norm = numpy.linalg.norm(c)
    assert c.shape == (N, N)
    assert_almost_equal(c_norm, C_NORM, decimal=10)
