import fermitools.oo.ocepa0 as ocepa0

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

NITER = 200
R_THRESH = 1e-13
H_ASO = numpy.load(os.path.join(data_path, 'h_aso.npy'))
G_ASO = numpy.load(os.path.join(data_path, 'g_aso.npy'))
HOV = numpy.load(os.path.join(data_path, 'ocepa0/hov.npy'))
GOOOO = numpy.load(os.path.join(data_path, 'ocepa0/goooo.npy'))
GOOOV = numpy.load(os.path.join(data_path, 'ocepa0/gooov.npy'))
GOOVV = numpy.load(os.path.join(data_path, 'ocepa0/goovv.npy'))
GOVOV = numpy.load(os.path.join(data_path, 'ocepa0/govov.npy'))
GOVVV = numpy.load(os.path.join(data_path, 'ocepa0/govvv.npy'))
GVVVV = numpy.load(os.path.join(data_path, 'ocepa0/gvvvv.npy'))
FOO = numpy.load(os.path.join(data_path, 'ocepa0/foo.npy'))
FOV = numpy.load(os.path.join(data_path, 'ocepa0/fov.npy'))
FVV = numpy.load(os.path.join(data_path, 'ocepa0/fvv.npy'))
T2 = numpy.load(os.path.join(data_path, 'ocepa0/t2.npy'))


def test__fock_xy():
    fov = ocepa0.fock_xy(HOV, GOOOV)
    assert_almost_equal(fov, FOV, decimal=10)


def test__twobody_amplitude_gradient():
    r2 = ocepa0.twobody_amplitude_gradient(
            GOOOO, GOOVV, GOVOV, GVVVV, FOO, FVV, T2)
    print(numpy.amax(r2))
    print(numpy.amin(r2))
    assert_almost_equal(r2, 0., decimal=10)


def test__orbital_gradient():
    r1 = ocepa0.orbital_gradient(FOV, GOOOV, GOVVV, T2)
    assert_almost_equal(r1, 0., decimal=10)
