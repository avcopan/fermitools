import fermitools.oo.odc12 as odc12

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

NITER = 200
R_THRESH = 1e-13
H_ASO = numpy.load(os.path.join(data_path, 'h_aso.npy'))
G_ASO = numpy.load(os.path.join(data_path, 'g_aso.npy'))
C_GUESS = numpy.load(os.path.join(data_path, 'c_guess.npy'))
T2_GUESS = numpy.load(os.path.join(data_path, 't2_guess.npy'))
HOO = numpy.load(os.path.join(data_path, 'odc12/hoo.npy'))
HOV = numpy.load(os.path.join(data_path, 'odc12/hov.npy'))
HVV = numpy.load(os.path.join(data_path, 'odc12/hvv.npy'))
GOOOO = numpy.load(os.path.join(data_path, 'odc12/goooo.npy'))
GOOOV = numpy.load(os.path.join(data_path, 'odc12/gooov.npy'))
GOOVV = numpy.load(os.path.join(data_path, 'odc12/goovv.npy'))
GOVOV = numpy.load(os.path.join(data_path, 'odc12/govov.npy'))
GOVVV = numpy.load(os.path.join(data_path, 'odc12/govvv.npy'))
GVVVV = numpy.load(os.path.join(data_path, 'odc12/gvvvv.npy'))
M1OO = numpy.load(os.path.join(data_path, 'odc12/m1oo.npy'))
M1VV = numpy.load(os.path.join(data_path, 'odc12/m1vv.npy'))
FOO = numpy.load(os.path.join(data_path, 'odc12/foo.npy'))
FOV = numpy.load(os.path.join(data_path, 'odc12/fov.npy'))
FVV = numpy.load(os.path.join(data_path, 'odc12/fvv.npy'))
POO = numpy.load(os.path.join(data_path, 'odc12/poo.npy'))
PVV = numpy.load(os.path.join(data_path, 'odc12/pvv.npy'))
FPOO = numpy.load(os.path.join(data_path, 'odc12/fpoo.npy'))
FFOO = numpy.load(os.path.join(data_path, 'odc12/ffoo.npy'))
FFVV = numpy.load(os.path.join(data_path, 'odc12/ffvv.npy'))
EN_ELEC = numpy.load(os.path.join(data_path, 'odc12/en_elec.npy'))
MU_ELEC = numpy.load(os.path.join(data_path, 'odc12/mu_elec.npy'))
C = numpy.load(os.path.join(data_path, 'odc12/c.npy'))
T2 = numpy.load(os.path.join(data_path, 'odc12/t2.npy'))


def test__fock_xy():
    fov = odc12.fock_xy(HOV, GOOOV, GOVVV, M1OO, M1VV)
    assert_almost_equal(fov, FOV, decimal=10)


def test__fancy_property():
    fpoo = odc12.fancy_property(POO, M1OO)
    assert_almost_equal(fpoo, FPOO, decimal=10)


def test__twobody_amplitude_gradient():
    r2 = odc12.twobody_amplitude_gradient(
            GOOOO, GOOVV, GOVOV, GVVVV, +FFOO, -FFVV, T2)
    print(numpy.amax(r2))
    print(numpy.amin(r2))
    assert_almost_equal(r2, 0., decimal=10)


def test__onebody_density():
    m1oo, m1vv = odc12.onebody_density(T2)
    assert_almost_equal(m1oo, M1OO, decimal=10)
    assert_almost_equal(m1vv, M1VV, decimal=10)


def test__orbital_gradient():
    r1 = odc12.orbital_gradient(
            FOV, GOOOV, GOVVV, M1OO, M1VV, T2)
    assert_almost_equal(r1, 0., decimal=10)


def test__electronic_energy():
    en_elec = odc12.electronic_energy(
            HOO, HVV, GOOOO, GOOVV, GOVOV, GVVVV, M1OO, M1VV, FOO, FVV, T2)
    assert_almost_equal(en_elec, EN_ELEC, decimal=10)


def test__solve():
    # test approximate guess
    en_elec, c, t2, info = odc12.solve(
            h_aso=H_ASO, g_aso=G_ASO, c_guess=C_GUESS, t2_guess=T2_GUESS,
            niter=NITER, r_thresh=R_THRESH)
    assert_almost_equal(en_elec, EN_ELEC, decimal=10)
    assert_almost_equal(c, C, decimal=10)
    assert_almost_equal(t2, T2, decimal=10)
    assert info['niter'] < 100
    assert info['r1_max'] < R_THRESH
    assert info['r2_max'] < R_THRESH
    # test perfect guess
    en_elec, c, t2, info = odc12.solve(
            h_aso=H_ASO, g_aso=G_ASO, c_guess=C, t2_guess=T2, niter=NITER,
            r_thresh=R_THRESH)
    assert_almost_equal(en_elec, EN_ELEC, decimal=10)
    assert_almost_equal(c, C, decimal=10)
    assert_almost_equal(t2, T2, decimal=8)
    assert info['niter'] == 1
    assert info['r1_max'] < R_THRESH
    assert info['r2_max'] < R_THRESH
    # test properties
    m1oo, m1vv = odc12.onebody_density(t2)
    mu_elec = [numpy.vdot(pxoo, m1oo) + numpy.vdot(pxvv, m1vv)
               for pxoo, pxvv in zip(POO, PVV)]
    assert_almost_equal(mu_elec, MU_ELEC, decimal=10)


if __name__ == '__main__':
    test__electronic_energy()
