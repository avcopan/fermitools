import fermitools.lr.odc12 as odc12

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
I1 = numpy.load(os.path.join(data_path, 'cation/i1.npy'))
I2 = numpy.load(os.path.join(data_path, 'cation/i2.npy'))
GOOOO = numpy.load(os.path.join(data_path, 'cation/odc12/goooo.npy'))
GOOOV = numpy.load(os.path.join(data_path, 'cation/odc12/gooov.npy'))
GOOVV = numpy.load(os.path.join(data_path, 'cation/odc12/goovv.npy'))
GOVOV = numpy.load(os.path.join(data_path, 'cation/odc12/govov.npy'))
GOVVV = numpy.load(os.path.join(data_path, 'cation/odc12/govvv.npy'))
GVVVV = numpy.load(os.path.join(data_path, 'cation/odc12/gvvvv.npy'))
T2 = numpy.load(os.path.join(data_path, 'cation/odc12/t2.npy'))
FFOO = numpy.load(os.path.join(data_path, 'cation/odc12/ffoo.npy'))
FFVV = numpy.load(os.path.join(data_path, 'cation/odc12/ffvv.npy'))
FGOOOO = numpy.load(os.path.join(data_path, 'cation/odc12/fgoooo.npy'))
FGOVOV = numpy.load(os.path.join(data_path, 'cation/odc12/fgovov.npy'))
FGVVVV = numpy.load(os.path.join(data_path, 'cation/odc12/fgvvvv.npy'))
FIOO = numpy.load(os.path.join(data_path, 'cation/odc12/fioo.npy'))
FIVV = numpy.load(os.path.join(data_path, 'cation/odc12/fivv.npy'))

A12 = numpy.load(os.path.join(data_path, 'cation/odc12/a12.npy'))
B12 = numpy.load(os.path.join(data_path, 'cation/odc12/b12.npy'))
A21 = numpy.load(os.path.join(data_path, 'cation/odc12/a21.npy'))
B21 = numpy.load(os.path.join(data_path, 'cation/odc12/b21.npy'))
A22 = numpy.load(os.path.join(data_path, 'cation/odc12/a22.npy'))
B22 = numpy.load(os.path.join(data_path, 'cation/odc12/b22.npy'))


def test__a12_sigma():
    a12_ = odc12.a12_sigma(GOOOV, GOVVV, FIOO, FIVV, T2)
    assert_almost_equal(a12_(I2), A12, decimal=12)


def test__b12_sigma():
    b12_ = odc12.b12_sigma(GOOOV, GOVVV, FIOO, FIVV, T2)
    assert_almost_equal(b12_(I2), B12, decimal=12)


def test__a21_sigma():
    a21_ = odc12.a21_sigma(GOOOV, GOVVV, FIOO, FIVV, T2)
    assert_almost_equal(a21_(I1), A21, decimal=12)


def test__b21_sigma():
    b21_ = odc12.b21_sigma(GOOOV, GOVVV, FIOO, FIVV, T2)
    assert_almost_equal(b21_(I1), B21, decimal=12)


def test__a22_sigma():
    a22_ = odc12.a22_sigma(
            FFOO, FFVV, GOOOO, GOVOV, GVVVV, FGOOOO, FGOVOV, FGVVVV, T2)
    assert_almost_equal(a22_(I2), A22, decimal=12)


def test__b22_sigma():
    b22_ = odc12.b22_sigma(FGOOOO, FGOVOV, FGVVVV, T2)
    assert_almost_equal(b22_(I2), B22, decimal=12)
