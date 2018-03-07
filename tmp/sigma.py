import fermitools.math.sigma as sigma

import numpy
import scipy
import time
import fermitools
from numpy.testing import assert_almost_equal

numpy.random.seed(0)


def m_sigma(dim):
    m_diag = numpy.arange(2, dim+2)
    rn = numpy.random.rand(dim)
    cp3 = .1 * numpy.pad(rn, (3, 0), mode='constant')[:-3, None]
    cp2 = .2 * numpy.pad(rn, (2, 0), mode='constant')[:-2, None]
    cp1 = .3 * numpy.pad(rn, (1, 0), mode='constant')[:-1, None]
    cp0 = m_diag[:, None]
    cm1 = .3 * numpy.pad(rn[:-1], (0, 1), mode='constant')[:, None]
    cm2 = .2 * numpy.pad(rn[:-2], (0, 2), mode='constant')[:, None]
    cm3 = .1 * numpy.pad(rn[:-3], (0, 3), mode='constant')[:, None]

    def _m(r):
        shape = r.shape
        r = r if r.ndim is 2 else numpy.reshape(r, (-1, 1))
        dm3 = numpy.pad(cm3 * r, ((3, 0), (0, 0)), mode='constant')[:-3]
        dm2 = numpy.pad(cm2 * r, ((2, 0), (0, 0)), mode='constant')[:-2]
        dm1 = numpy.pad(cm1 * r, ((1, 0), (0, 0)), mode='constant')[:-1]
        dp0 = r * cp0
        dp1 = numpy.pad(cp1 * r, ((0, 1), (0, 0)), mode='constant')[1:]
        dp2 = numpy.pad(cp2 * r, ((0, 2), (0, 0)), mode='constant')[2:]
        dp3 = numpy.pad(cp3 * r, ((0, 3), (0, 0)), mode='constant')[3:]
        ar = dm3 + dm2 + dm1 + dp0 + dp1 + dp2 + dp3
        return numpy.reshape(ar, shape)

    return _m


dim = 2000
neig = 4
nguess = 5
nsvec = 2
nvec = 15
rthresh = 1e-11
a_ = m_sigma(dim)
b_ = fermitools.math.sigma.eye

T0 = time.time()
a = a_(numpy.eye(dim))
vals, vecs = scipy.linalg.eigh(a)
DT = time.time() - T0
W = vals[:neig]
U = vecs[:, :neig]

ad = fermitools.math.sigma.diagonal(a_, dim)
bd = fermitools.math.sigma.diagonal(b_, dim)
guess = fermitools.math.sigma.evec_guess(ad, nguess)
w, u, info = sigma.eighg(
        a=a_, b=b_, neig=neig, ad=ad, bd=bd, guess=guess, rthresh=rthresh,
        nsvec=nsvec, nvec=nvec, disk=True)
assert_almost_equal(w, W, decimal=10)
assert_almost_equal(numpy.abs(u), numpy.abs(U), decimal=10)
