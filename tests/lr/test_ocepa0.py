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
GOOVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/goovv.npy'))
GOVOV = numpy.load(os.path.join(data_path, 'cation/ocepa0/govov.npy'))
GVVVV = numpy.load(os.path.join(data_path, 'cation/ocepa0/gvvvv.npy'))
FOO = numpy.load(os.path.join(data_path, 'cation/ocepa0/foo.npy'))
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
A11 = numpy.load(os.path.join(data_path, 'cation/ocepa0/a11.npy'))
B11 = numpy.load(os.path.join(data_path, 'cation/ocepa0/b11.npy'))
S11 = numpy.load(os.path.join(data_path, 'cation/ocepa0/s11.npy'))
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
    s11_ = ocepa0.s11_sigma(M1OO, M1VV)
    assert_almost_equal(s11_(I1), S11, decimal=12)


def test__a11_sigma():
    a11_ = ocepa0.a11_sigma(
            HOO, HVV, GOOOO, GOOVV, GOVOV, GVVVV, M1OO, M1VV, M2OOOO, M2OOVV,
            M2OVOV, M2VVVV)
    assert_almost_equal(a11_(I1), A11, decimal=12)


def test__b11_sigma():
    b11_ = ocepa0.b11_sigma(
            GOOOO, GOOVV, GOVOV, GVVVV, M2OOOO, M2OOVV, M2OVOV, M2VVVV)
    assert_almost_equal(b11_(I1), B11, decimal=12)


def test__a22_sigma():
    a22_ = ocepa0.a22_sigma(FOO, FVV, GOOOO, GOVOV, GVVVV)
    assert_almost_equal(a22_(I2), A22, decimal=12)
