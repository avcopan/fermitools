from fermitools.chem import nuc

from numpy.testing import assert_almost_equal

LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

ENERGY = 8.00236706181
DIPOLE = (0.0, 0.0, 1.1272911126779999)


def test__energy():
    energy = nuc.energy(labels=LABELS, coords=COORDS)
    assert_almost_equal(energy, ENERGY, decimal=10)


def test__electric_dipole():
    dipole = nuc.electric_dipole(labels=LABELS, coords=COORDS)
    assert_almost_equal(dipole, DIPOLE, decimal=10)
