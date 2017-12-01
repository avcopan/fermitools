import fermitools.lr.hf as hf

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
NO = numpy.load(os.path.join(data_path, 'neutral/no.npy'))
NV = numpy.load(os.path.join(data_path, 'neutral/nv.npy'))
POV = numpy.load(os.path.join(data_path, 'neutral/hf/pov.npy'))
FOO = numpy.load(os.path.join(data_path, 'neutral/hf/foo.npy'))
FVV = numpy.load(os.path.join(data_path, 'neutral/hf/fvv.npy'))
GOVOV = numpy.load(os.path.join(data_path, 'neutral/hf/govov.npy'))
GOOVV = numpy.load(os.path.join(data_path, 'neutral/hf/goovv.npy'))
PG = numpy.load(os.path.join(data_path, 'neutral/hf/t.npy'))
A = numpy.load(os.path.join(data_path, 'neutral/hf/a.npy'))
B = numpy.load(os.path.join(data_path, 'neutral/hf/b.npy'))


def test__pg():
    pg = hf.pg(POV)
    assert_almost_equal(pg, PG, decimal=12)


def test__a_sigma():
    i = numpy.reshape(numpy.eye(NO * NV), (NO, NV, NO, NV))
    a_ = hf.a_sigma(FOO, FVV, GOVOV)
    assert_almost_equal(a_(i), A, decimal=12)


def test__b_sigma():
    i = numpy.reshape(numpy.eye(NO * NV), (NO, NV, NO, NV))
    b_ = hf.b_sigma(GOOVV)
    assert_almost_equal(b_(i), B, decimal=12)
