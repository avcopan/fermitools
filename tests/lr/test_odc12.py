import fermitools.lr.odc12 as odc12

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

NO = numpy.load(os.path.join(data_path, 'no.npy'))
NV = numpy.load(os.path.join(data_path, 'nv.npy'))
POO = numpy.load(os.path.join(data_path, 'odc12/poo.npy'))
POV = numpy.load(os.path.join(data_path, 'odc12/pov.npy'))
PVV = numpy.load(os.path.join(data_path, 'odc12/pvv.npy'))
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
CFOO = numpy.load(os.path.join(data_path, 'odc12/cfoo.npy'))
CFVV = numpy.load(os.path.join(data_path, 'odc12/cfvv.npy'))
FFOO = numpy.load(os.path.join(data_path, 'odc12/ffoo.npy'))
FFVV = numpy.load(os.path.join(data_path, 'odc12/ffvv.npy'))
FIOO = numpy.load(os.path.join(data_path, 'odc12/fioo.npy'))
FIVV = numpy.load(os.path.join(data_path, 'odc12/fivv.npy'))
FPOO = numpy.load(os.path.join(data_path, 'odc12/fpoo.npy'))
FPVV = numpy.load(os.path.join(data_path, 'odc12/fpvv.npy'))
FGOOOO = numpy.load(os.path.join(data_path, 'odc12/fgoooo.npy'))
FGOVOV = numpy.load(os.path.join(data_path, 'odc12/fgovov.npy'))
FGVVVV = numpy.load(os.path.join(data_path, 'odc12/fgvvvv.npy'))
T2 = numpy.load(os.path.join(data_path, 'odc12/t2.npy'))
PG1 = -numpy.load(os.path.join(data_path, 'odc12/pg1.npy'))
PG2 = numpy.load(os.path.join(data_path, 'odc12/pg2.npy'))
PG = numpy.load(os.path.join(data_path, 'odc12/pg.npy'))
EYE = numpy.load(os.path.join(data_path, 'eye.npy'))
I1U = numpy.load(os.path.join(data_path, 'i1u.npy'))
I2U = numpy.load(os.path.join(data_path, 'i2u.npy'))
AD1 = numpy.load(os.path.join(data_path, 'odc12/ad1.npy'))
AD2 = numpy.load(os.path.join(data_path, 'odc12/ad2.npy'))
A11 = numpy.load(os.path.join(data_path, 'odc12/a11.npy'))
B11 = numpy.load(os.path.join(data_path, 'odc12/b11.npy'))
A12 = -numpy.load(os.path.join(data_path, 'odc12/a12.npy'))
A21 = -numpy.load(os.path.join(data_path, 'odc12/a21.npy'))
B12 = -numpy.load(os.path.join(data_path, 'odc12/b12.npy'))
B21 = -numpy.load(os.path.join(data_path, 'odc12/b21.npy'))
A22 = numpy.load(os.path.join(data_path, 'odc12/a22.npy'))
B22 = numpy.load(os.path.join(data_path, 'odc12/b22.npy'))
S11 = numpy.load(os.path.join(data_path, 'odc12/s11.npy'))
AD = numpy.load(os.path.join(data_path, 'odc12/ad.npy'))
SD = numpy.load(os.path.join(data_path, 'odc12/sd.npy'))
A = numpy.load(os.path.join(data_path, 'odc12/a.npy'))
B = numpy.load(os.path.join(data_path, 'odc12/b.npy'))
S = numpy.load(os.path.join(data_path, 'odc12/s.npy'))
D = numpy.load(os.path.join(data_path, 'odc12/d.npy'))


def test__onebody_hessian_zeroth_order_diagonal():
    ad1 = odc12.onebody_hessian_zeroth_order_diagonal(FOO, FVV)
    assert_almost_equal(ad1, AD1, decimal=10)


def test__twobody_hessian_zeroth_order_diagonal():
    ad2 = odc12.twobody_hessian_zeroth_order_diagonal(FFOO, FFVV)
    assert_almost_equal(ad2, AD2, decimal=10)


def test__onebody_property_gradient():
    pg1 = odc12.onebody_property_gradient(POV, M1OO, M1VV)
    assert_almost_equal(pg1, PG1, decimal=10)


def test__twobody_property_gradient():
    pg2 = odc12.twobody_property_gradient(FPOO, FPVV, T2)
    assert_almost_equal(pg2, PG2, decimal=10)


def test__onebody_hessian():
    a11, b11 = odc12.onebody_hessian(
            FOO, FVV, CFOO, CFVV, GOOOO, GOOVV, GOVOV, GVVVV, T2, M1OO, M1VV)
    assert_almost_equal(a11(I1U), A11, decimal=10)
    assert_almost_equal(b11(I1U), B11, decimal=10)


def test__mixed_hessian():
    a12, b12, a21, b21 = odc12.mixed_hessian(FIOO, FIVV, GOOOV, GOVVV, T2)
    assert_almost_equal(a12(I2U), A12, decimal=10)
    assert_almost_equal(b12(I2U), B12, decimal=10)
    assert_almost_equal(a21(I1U), A21, decimal=10)
    assert_almost_equal(b21(I1U), B21, decimal=10)


def test__twobody_hessian():
    a22, b22 = odc12.twobody_hessian(
            FFOO, FFVV, GOOOO, GOVOV, GVVVV, FGOOOO, FGOVOV, FGVVVV, T2)
    assert_almost_equal(a22(I2U), A22, decimal=10)
    assert_almost_equal(b22(I2U), B22, decimal=10)


def test__onebody_metric():
    s11 = odc12.onebody_metric(T2)
    assert_almost_equal(s11(I1U), S11, decimal=10)


def test__onebody_metric_function():
    x11 = odc12.onebody_metric_function(T2, f=numpy.reciprocal)
    assert_almost_equal(x11(S11), I1U, decimal=10)


if __name__ == '__main__':
    test__twobody_hessian()
