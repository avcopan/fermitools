import simplehf.interface.pyscf as interface

import numpy
from numpy.testing import assert_almost_equal

N = 7
BASIS = 'STO-3G'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

OVERLAP_NORM = 2.87134489728
KINETIC_NORM = 29.3665362237
NUCLEAR_NORM = 66.4857540061
HCORE_NORM = 39.0837944394
DIPOLE_NORM = 3.54727809028
REPULSION_NORM = 7.77961419893


def test__overlap():
    overlap = interface.integrals.overlap(basis=BASIS, labels=LABELS,
                                          coords=COORDS)
    overlap_norm = numpy.linalg.norm(overlap)

    assert overlap.shape == (N, N)
    assert_almost_equal(overlap_norm, OVERLAP_NORM, decimal=10)


def test__kinetic():
    kinetic = interface.integrals.kinetic(basis=BASIS, labels=LABELS,
                                          coords=COORDS)
    kinetic_norm = numpy.linalg.norm(kinetic)

    assert kinetic.shape == (N, N)
    assert_almost_equal(kinetic_norm, KINETIC_NORM, decimal=10)


def test__nuclear():
    nuclear = interface.integrals.nuclear(basis=BASIS, labels=LABELS,
                                          coords=COORDS)
    nuclear_norm = numpy.linalg.norm(nuclear)

    assert nuclear.shape == (N, N)
    assert_almost_equal(nuclear_norm, NUCLEAR_NORM, decimal=10)


def test__core_hamiltonian():
    hcore = interface.integrals.core_hamiltonian(basis=BASIS, labels=LABELS,
                                                 coords=COORDS)
    hcore_norm = numpy.linalg.norm(hcore)

    assert hcore.shape == (N, N)
    assert_almost_equal(hcore_norm, HCORE_NORM, decimal=10)


def test__dipole():
    dipole = interface.integrals.dipole(basis=BASIS, labels=LABELS,
                                        coords=COORDS)
    dipole_norm = numpy.linalg.norm(dipole)

    assert dipole.shape == (3, N, N)
    assert_almost_equal(dipole_norm, DIPOLE_NORM, decimal=10)


def test__repulsion():
    repulsion = interface.integrals.repulsion(basis=BASIS, labels=LABELS,
                                              coords=COORDS)
    repulsion_norm = numpy.linalg.norm(repulsion)

    assert repulsion.shape == (N, N, N, N)
    assert_almost_equal(repulsion_norm, REPULSION_NORM, decimal=10)
