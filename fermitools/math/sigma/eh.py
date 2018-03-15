import numpy
import itertools
import scipy.linalg
from toolz import itertoolz
from more_itertools import consume

from .util import evec_guess
from ..disk import dataset, remove_dataset

import sys
import warnings


def eighg(a, b, neig, ad, bd, nguess=None, niter=100, nsvec=100, nvec=100,
          rthresh=1e-5, print_conv=True, highest=False, guess_type=None,
          disk=False):
    roots = slice(None, neig) if not highest else slice(None, -neig-1, -1)

    ns = ()
    vs = ()
    avs = ()
    bvs = ()

    a_blks = {}
    b_blks = {}

    nguess = 2 * neig if nguess is None else nguess
    assert nguess >= neig

    dim = len(ad)

    if isinstance(guess_type, numpy.ndarray):
        guess = guess_type
    elif guess_type is None:
        guess = evec_guess(ad, nguess, bd=bd, highest=highest)
    elif guess_type == 'random':
        rand = numpy.random.random((dim, nguess))
        guess, s, _ = scipy.linalg.svd(rand, full_matrices=False,
                                       overwrite_a=True)
    else:
        print('Guess type {:s} is not implemented.'.format(str(guess_type)))

    v = guess

    for iteration in range(niter):
        _, n = v.shape
        stops = tuple(range(0, n, nsvec)) + (n,)

        new_vs = ()
        for start, end in itertoolz.sliding_window(2, stops):
            vi = v[:, start:end]
            _, ni = numpy.shape(vi)
            vi = dataset(vi) if disk else vi
            new_vs += (vi,)
            ns += (ni,)
            vs += (vi,)

        for s, vi in enumerate(new_vs):
            vi = numpy.array(vi)
            avi = dataset(a(vi)) if disk else a(vi)
            bvi = dataset(b(vi)) if disk else b(vi)

            avs += (avi,)
            bvs += (bvi,)

            i = len(avs) - 1
            for j, vj in enumerate(vs):
                a_blks[i, j] = numpy.dot(numpy.transpose(avi), vj)
                b_blks[i, j] = numpy.dot(numpy.transpose(bvi), vj)
                a_blks[j, i] = numpy.transpose(a_blks[i, j])
                b_blks[j, i] = numpy.transpose(b_blks[i, j])

            print("subiteration {:d}".format(s))
            sys.stdout.flush()

        nblks = len(ns)
        a_red = numpy.bmat(
                [[a_blks[i, j] for j in range(nblks)] for i in range(nblks)])
        b_red = numpy.bmat(
                [[b_blks[i, j] for j in range(nblks)] for i in range(nblks)])
        vals, vecs = scipy.linalg.eigh(a=a_red, b=b_red)

        w = vals[roots]
        y = vecs[:, roots]
        blks = tuple(itertools.accumulate(ns))
        ys = numpy.split(y, blks, axis=0)

        x = sum(numpy.dot(vj, yj) for vj, yj in zip(vs, ys))
        ax = sum(numpy.dot(avj, yj) for avj, yj in zip(avs, ys))
        bx = sum(numpy.dot(bvj, yj) for bvj, yj in zip(bvs, ys))

        r = ax - bx * w
        rmax = numpy.amax(numpy.abs(r))

        rdim = sum(ns)

        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        converged = rmax < rthresh

        if print_conv:
            print(info)
            print(1/w)
            sys.stdout.flush()

        if converged:
            break

        precnd = numpy.reshape(w[None, :] * bd[:, None] - ad[:, None], r.shape)
        v = r / precnd
        for vi in vs:
            vi = numpy.array(vi)
            v -= numpy.linalg.multi_dot([vi, vi.T, v])
        v, s, _ = scipy.linalg.svd(v, full_matrices=False, overwrite_a=True)
        tol = max(v.shape) * numpy.amax(numpy.abs(v)) * numpy.finfo(float).eps
        v = v[:, s > tol]

        if rdim + v.shape[1] > nvec:
            if disk:
                consume(map(remove_dataset, vs))
                consume(map(remove_dataset, avs))
                consume(map(remove_dataset, bvs))

            ns = ()
            vs = ()
            avs = ()
            bvs = ()

            a_blks = {}
            b_blks = {}

            v = x

    if disk:
        consume(map(remove_dataset, vs))
        consume(map(remove_dataset, avs))
        consume(map(remove_dataset, bvs))

    if not converged:
        warnings.warn("Did not converge! (rmax: {:7.1e})".format(rmax))

    return w, x, info
