import fermitools.lr.ocepa0 as ocepa0

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

POO = numpy.load(os.path.join(data_path, 'ocepa0/poo.npy'))
POV = numpy.load(os.path.join(data_path, 'ocepa0/pov.npy'))
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
T2 = numpy.load(os.path.join(data_path, 'ocepa0/t2.npy'))
PG1 = numpy.load(os.path.join(data_path, 'ocepa0/pg1.npy'))
PG2 = numpy.load(os.path.join(data_path, 'ocepa0/pg2.npy'))
I1U = numpy.load(os.path.join(data_path, 'ocepa0/i1u.npy'))
I2U = numpy.load(os.path.join(data_path, 'ocepa0/i2u.npy'))
S11 = numpy.load(os.path.join(data_path, 'ocepa0/s11.npy'))
A11 = numpy.load(os.path.join(data_path, 'ocepa0/a11.npy'))
B11 = numpy.load(os.path.join(data_path, 'ocepa0/b11.npy'))
A12 = numpy.load(os.path.join(data_path, 'ocepa0/a12.npy'))
B12 = numpy.load(os.path.join(data_path, 'ocepa0/b12.npy'))
A21 = numpy.load(os.path.join(data_path, 'ocepa0/a21.npy'))
B21 = numpy.load(os.path.join(data_path, 'ocepa0/b21.npy'))
A22 = numpy.load(os.path.join(data_path, 'ocepa0/a22.npy'))


def test__onebody_property_gradient():
    pg1 = ocepa0.onebody_property_gradient(POV, T2)
    assert_almost_equal(pg1, PG1, decimal=12)


def test__twobody_property_gradient():
    pg2 = ocepa0.twobody_property_gradient(POO, PVV, T2)
    assert_almost_equal(pg2, PG2, decimal=12)


def test__onebody_hessian():
    a11, b11 = ocepa0.onebody_hessian(
            FOO, FVV, GOOOO, GOOVV, GOVOV, GVVVV, T2)
    assert_almost_equal(a11(I1U), A11, decimal=12)
    assert_almost_equal(b11(I1U), B11, decimal=12)


def test__mixed_hessian():
    a12, b12, a21, b21 = ocepa0.mixed_hessian(FOV, GOOOV, GOVVV, T2)
    assert_almost_equal(a12(I2U), A12, decimal=10)
    assert_almost_equal(b12(I2U), B12, decimal=10)
    assert_almost_equal(a21(I1U), A21, decimal=10)
    assert_almost_equal(b21(I1U), B21, decimal=10)


def test__twobody_hessian():
    a22 = ocepa0.twobody_hessian(
            FOO, FVV, GOOOO, GOVOV, GVVVV)
    assert_almost_equal(a22(I2U), A22, decimal=10)


def test__onebody_metric():
    s11 = ocepa0.onebody_metric(T2)
    assert_almost_equal(s11(I1U), S11, decimal=10)
