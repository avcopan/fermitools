import fermitools.lr.ocepa0 as ocepa0

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
I1 = numpy.load(os.path.join(data_path, 'cation/i1.npy'))
I2 = numpy.load(os.path.join(data_path, 'cation/i2.npy'))
HOO = numpy.load(os.path.join(data_path, 'cation/ocepa0/hoo.npy'))
HVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/hvv.npy'))
POO = numpy.load(os.path.join(data_path, 'cation/ocepa0/poo.npy'))
POV = numpy.load(os.path.join(data_path, 'cation/ocepa0/pov.npy'))
PVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/pvv.npy'))
GOOOO = numpy.load(os.path.join(data_path, 'cation/ocepa0/goooo.npy'))
GOOOV = numpy.load(os.path.join(data_path, 'cation/ocepa0/gooov.npy'))
GOOVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/goovv.npy'))
GOVOV = numpy.load(os.path.join(data_path, 'cation/ocepa0/govov.npy'))
GOVVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/govvv.npy'))
GVVVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/gvvvv.npy'))
FOO = numpy.load(os.path.join(data_path, 'cation/ocepa0/foo.npy'))
FOV = numpy.load(os.path.join(data_path, 'cation/ocepa0/fov.npy'))
FVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/fvv.npy'))
T2 = numpy.load(os.path.join(data_path, 'cation/ocepa0/t2.npy'))
M1OO = numpy.load(os.path.join(data_path, 'cation/ocepa0/m1oo.npy'))
M1VV = numpy.load(os.path.join(data_path, 'cation/ocepa0/m1vv.npy'))
M2OOOO = numpy.load(os.path.join(data_path, 'cation/ocepa0/m2oooo.npy'))
M2OOVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/m2oovv.npy'))
M2OVOV = numpy.load(os.path.join(data_path, 'cation/ocepa0/m2ovov.npy'))
M2VVVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/m2vvvv.npy'))

S11_MAT = numpy.load(os.path.join(data_path, 'cation/ocepa0/s11_mat.npy'))
PG1 = numpy.load(os.path.join(data_path, 'cation/ocepa0/pg1.npy'))
PG2 = numpy.load(os.path.join(data_path, 'cation/ocepa0/pg2.npy'))
S11 = numpy.load(os.path.join(data_path, 'cation/ocepa0/s11.npy'))
A11 = numpy.load(os.path.join(data_path, 'cation/ocepa0/a11.npy'))
B11 = numpy.load(os.path.join(data_path, 'cation/ocepa0/b11.npy'))
A12 = numpy.load(os.path.join(data_path, 'cation/ocepa0/a12.npy'))
B12 = numpy.load(os.path.join(data_path, 'cation/ocepa0/b12.npy'))
A21 = numpy.load(os.path.join(data_path, 'cation/ocepa0/a21.npy'))
B21 = numpy.load(os.path.join(data_path, 'cation/ocepa0/b21.npy'))
A22 = numpy.load(os.path.join(data_path, 'cation/ocepa0/a22.npy'))


def test__s11_matrix():
    s11_mat = ocepa0.s11_matrix(M1OO, M1VV)
    assert_almost_equal(s11_mat, S11_MAT, decimal=12)


def test__onebody_property_gradient():
    pg1 = ocepa0.onebody_property_gradient(POV, M1OO, M1VV)
    assert_almost_equal(pg1, PG1, decimal=12)


def test__twobody_property_gradient():
    pg2 = ocepa0.twobody_property_gradient(POO, PVV, T2)
    assert_almost_equal(pg2, PG2, decimal=12)


def test__s11_sigma():
    s11 = ocepa0.s11_sigma(M1OO, M1VV)
    assert_almost_equal(s11(I1), S11, decimal=12)


def test__a11_sigma():
    a11 = ocepa0.a11_sigma(
            HOO, HVV, GOOOO, GOOVV, GOVOV, GVVVV, M1OO, M1VV, M2OOOO, M2OOVV,
            M2OVOV, M2VVVV)
    assert_almost_equal(a11(I1), A11, decimal=12)


def test__b11_sigma():
    b11 = ocepa0.b11_sigma(
            GOOOO, GOOVV, GOVOV, GVVVV, M2OOOO, M2OOVV, M2OVOV, M2VVVV)
    assert_almost_equal(b11(I1), B11, decimal=12)


def test__a12_sigma():
    a12 = ocepa0.a12_sigma(FOV, GOOOV, GOVVV, T2)
    assert_almost_equal(a12(I2), A12, decimal=12)


def test__b12_sigma():
    b12 = ocepa0.b12_sigma(FOV, GOOOV, GOVVV, T2)
    assert_almost_equal(b12(I2), B12, decimal=12)


def test__a21_sigma():
    a21 = ocepa0.a21_sigma(FOV, GOOOV, GOVVV, T2)
    assert_almost_equal(a21(I1), A21, decimal=12)


def test__b21_sigma():
    b21 = ocepa0.b21_sigma(FOV, GOOOV, GOVVV, T2)
    assert_almost_equal(b21(I1), B21, decimal=12)


def test__a22_sigma():
    a22 = ocepa0.a22_sigma(FOO, FVV, GOOOO, GOVOV, GVVVV)
    assert_almost_equal(a22(I2), A22, decimal=12)
