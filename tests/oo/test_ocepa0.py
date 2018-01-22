import fermitools.oo.ocepa0 as ocepa0

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

NITER = 200
R_THRESH = 1e-13
NA = numpy.load(os.path.join(data_path, 'na.npy'))
NB = numpy.load(os.path.join(data_path, 'nb.npy'))
H_AO = numpy.load(os.path.join(data_path, 'h_ao.npy'))
R_AO = numpy.load(os.path.join(data_path, 'r_ao.npy'))
C_GUESS = numpy.load(os.path.join(data_path, 'c_guess.npy'))
T2_GUESS = numpy.load(os.path.join(data_path, 't2_guess.npy'))
HOO = numpy.load(os.path.join(data_path, 'ocepa0/hoo.npy'))
HOV = numpy.load(os.path.join(data_path, 'ocepa0/hov.npy'))
HVV = numpy.load(os.path.join(data_path, 'ocepa0/hvv.npy'))
POO = numpy.load(os.path.join(data_path, 'ocepa0/poo.npy'))
PVV = numpy.load(os.path.join(data_path, 'ocepa0/pvv.npy'))
GOOOO = numpy.load(os.path.join(data_path, 'ocepa0/goooo.npy'))
GOOOV = numpy.load(os.path.join(data_path, 'ocepa0/gooov.npy'))
GOOVV = numpy.load(os.path.join(data_path, 'ocepa0/goovv.npy'))
GOVOV = numpy.load(os.path.join(data_path, 'ocepa0/govov.npy'))
GOVVV = numpy.load(os.path.join(data_path, 'ocepa0/govvv.npy'))
GVVVV = numpy.load(os.path.join(data_path, 'ocepa0/gvvvv.npy'))
FOO = numpy.load(os.path.join(data_path, 'ocepa0/foo.npy'))
FOV = numpy.load(os.path.join(data_path, 'ocepa0/fov.npy'))
FVV = numpy.load(os.path.join(data_path, 'ocepa0/fvv.npy'))
C = numpy.load(os.path.join(data_path, 'ocepa0/c.npy'))
T2 = numpy.load(os.path.join(data_path, 'ocepa0/t2.npy'))
M1OO = numpy.load(os.path.join(data_path, 'ocepa0/m1oo.npy'))
M1VV = numpy.load(os.path.join(data_path, 'ocepa0/m1vv.npy'))
EN_ELEC = numpy.load(os.path.join(data_path, 'ocepa0/en_elec.npy'))
MU_ELEC = numpy.load(os.path.join(data_path, 'ocepa0/mu_elec.npy'))


def test__solve():
    # test approximate guess
    en_elec, c, t2, info = ocepa0.solve(
            NA, NB, H_AO, R_AO, C_GUESS, T2_GUESS, NITER, R_THRESH, True)
    assert_almost_equal(en_elec, EN_ELEC, decimal=10)
    assert_almost_equal(c, C, decimal=10)
    assert_almost_equal(t2, T2, decimal=10)
    assert info['niter'] < 150
    assert info['r1_max'] < R_THRESH
    assert info['r2_max'] < R_THRESH
    # test perfect guess
    en_elec, c, t2, info = ocepa0.solve(
            NA, NB, H_AO, R_AO, C, T2, NITER, R_THRESH, True)
    assert_almost_equal(en_elec, EN_ELEC, decimal=10)
    assert_almost_equal(c, C, decimal=10)
    assert_almost_equal(t2, T2, decimal=8)
    assert info['niter'] == 1
    assert info['r1_max'] < R_THRESH
    assert info['r2_max'] < R_THRESH
    # test properties
    m1oo, m1vv = ocepa0.onebody_density(t2)
    mu_elec = [numpy.vdot(pxoo, m1oo) + numpy.vdot(pxvv, m1vv)
               for pxoo, pxvv in zip(POO, PVV)]
    assert_almost_equal(mu_elec, MU_ELEC, decimal=10)


def test__fock_xy():
    fov = ocepa0.fock_xy(HOV, GOOOV)
    assert_almost_equal(fov, FOV, decimal=10)


def test__twobody_amplitude_gradient():
    r2 = ocepa0.twobody_amplitude_gradient(
            GOOOO, GOOVV, GOVOV, GVVVV, FOO, FVV, T2)
    assert_almost_equal(r2, 0., decimal=10)


def test__orbital_gradient():
    r1 = ocepa0.orbital_gradient(FOV, GOOOV, GOVVV, T2)
    assert_almost_equal(r1, 0., decimal=10)


def test__electronic_energy():
    en_elec = ocepa0.electronic_energy(
            HOO, HVV, GOOOO, GOOVV, GOVOV, GVVVV, T2)
    assert_almost_equal(en_elec, EN_ELEC, decimal=10)


def test__onebody_density():
    m1oo, m1vv = ocepa0.onebody_density(T2)
    assert_almost_equal(m1oo, M1OO, decimal=10)
    assert_almost_equal(m1vv, M1VV, decimal=10)


if __name__ == '__main__':
    test__solve()
