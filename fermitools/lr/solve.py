import numpy
from toolz import functoolz
from ..math.sigma import add
from ..math.sigma import subtract
from ..math.sigma import bmat
from ..math.sigma import negative
from ..math.sigma import eighg
from ..math.sigma import solve


def static_response(a, b, pg, ad, nvec=100, niter=50, rthresh=1e-5):
    e = add(a, b)
    v = -2*pg
    guess = v / ad[:, None]

    r, info = solve(
            a=e, b=v, ad=ad, guess=guess, niter=niter, nvec=nvec,
            rthresh=rthresh)

    return r, info


def spectrum(a, b, s, d, ad, sd, nroot=1, nguess=10, nsvec=10, nvec=100,
             niter=50, rthresh=1e-7, guess_random=False, disk=False,
             x=None):
    e = bmat([[a, b], [b, a]], 2)
    m = bmat([[s, d], [negative(d), negative(s)]], 2)
    ed = numpy.concatenate((+ad, +ad))
    md = numpy.concatenate((+sd, -sd))

    from ..math.direct import eig, eig_simple

    h = functoolz.compose(x, add(a, b), x, subtract(a, b))
    hd = ad * ad
    eye = (lambda x: x)
    eyed = numpy.ones_like(hd)

    eig(a=h, k=nroot, ad=hd, nconv=None, nx0=nguess*nroot, highest=False,
        maxvecs=nvec*nroot, maxiter=niter, tol=rthresh, print_conv=True)

    print('simple:')
    eig_simple(
            a=h, b=eye, neig=nroot, ad=hd, bd=eyed, nguess=nguess*nroot,
            rthresh=rthresh, nvec=nvec*nroot, niter=niter, highest=False,
            print_conv=True)

    hmat = h(numpy.eye(len(hd)))

    import scipy.linalg
    w2, x, y = scipy.linalg.eig(hmat, left=True, right=True)
    print(numpy.real(numpy.sqrt(w2)))

    from ..math.direct import eigh
    eigh(a=m, k=nroot, ad=md, b=e, bd=ed, nconv=None, nx0=nguess*nroot,
         highest=True, maxvecs=nvec*nroot, maxiter=niter, tol=rthresh,
         print_conv=True)

    w_inv, z, info = eighg(
            a=m, b=e, neig=nroot, ad=md, bd=ed, nguess=nguess*nroot,
            rthresh=rthresh, nsvec=nsvec, nvec=nvec*nroot, niter=niter,
            highest=True, guess_random=guess_random, disk=disk,
            print_conv=False)
    print(info)
    w = 1. / w_inv
    x, y = numpy.split(z, 2)
    return w, x, y, info
