import fermitools.lr.odc12 as odc12

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
I1 = numpy.load(os.path.join(data_path, 'cation/i1.npy'))
I2 = numpy.load(os.path.join(data_path, 'cation/i2.npy'))
GOOOO = numpy.load(os.path.join(data_path, 'cation/odc12/goooo.npy'))
GOOVV = numpy.load(os.path.join(data_path, 'cation/odc12/goovv.npy'))
GOVOV = numpy.load(os.path.join(data_path, 'cation/odc12/govov.npy'))
GVVVV = numpy.load(os.path.join(data_path, 'cation/odc12/gvvvv.npy'))
T2 = numpy.load(os.path.join(data_path, 'cation/odc12/t2.npy'))
FFOO = numpy.load(os.path.join(data_path, 'cation/odc12/ffoo.npy'))
FFVV = numpy.load(os.path.join(data_path, 'cation/odc12/ffvv.npy'))
FGOOOO = numpy.load(os.path.join(data_path, 'cation/odc12/fgoooo.npy'))
FGOVOV = numpy.load(os.path.join(data_path, 'cation/odc12/fgovov.npy'))
FGVVVV = numpy.load(os.path.join(data_path, 'cation/odc12/fgvvvv.npy'))

A22 = numpy.load(os.path.join(data_path, 'cation/odc12/a22.npy'))
B22 = numpy.load(os.path.join(data_path, 'cation/odc12/b22.npy'))


def test__a22_sigma():
    a22_ = odc12.a22_sigma(
            FFOO, FFVV, GOOOO, GOVOV, GVVVV, FGOOOO, FGOVOV, FGVVVV, T2)
    assert_almost_equal(a22_(I2), A22, decimal=12)


def test__b22_sigma():
    b22_ = odc12.b22_sigma(FGOOOO, FGOVOV, FGVVVV, T2)
    assert_almost_equal(b22_(I2), B22, decimal=12)
