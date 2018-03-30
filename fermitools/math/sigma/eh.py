import numpy
import itertools
import scipy.linalg
from toolz import itertoolz
from more_itertools import consume

from ..disk import dataset, remove_dataset, empty_dataset

import sys
import warnings


def iterate_block_boundaries(dim, block_size):
    dividers = tuple(range(0, dim, block_size)) + (dim,)
    return itertoolz.sliding_window(2, dividers)


def eighg(a, b, neig, ad, bd, nguess=None, niter=100, nsvec=100, nvec=100,
          rthresh=1e-5, print_conv=True, highest=False, guess_random=False,
          disk=False):
    roots = slice(None, neig) if not highest else slice(None, -neig-1, -1)

    ns = ()
    vs = ()
    avs = ()
    bvs = ()

    a_blks = {}
    b_blks = {}

    nguess = neig if nguess is None else nguess
    assert nguess >= neig

    dim = len(ad)

    ixs = numpy.argsort(ad / bd)[::-1] if highest else numpy.argsort(ad / bd)

    new_vs = ()
    new_ns = ()

    for start, end in iterate_block_boundaries(nguess, nsvec):
        ni = end-start
        if not guess_random:
            keys = (ixs[start:end], range(ni))
            vals = numpy.ones(ni)
            vi = scipy.sparse.coo_matrix((vals, keys),
                                         shape=(dim, ni)).toarray()
            vi = dataset(vi) if disk else vi
        else:
            vi = numpy.random.random((dim, ni))
            for vj in new_vs:
                vj = numpy.array(vj)
                vi -= numpy.linalg.multi_dot([vj, vj.T, vi])
            vi, _, _ = scipy.linalg.svd(vi, full_matrices=False,
                                        overwrite_a=True)
            vi = dataset(vi) if disk else vi
        new_vs += (vi,)
        new_ns += (ni,)

    for iteration in range(niter):
        vs += new_vs
        ns += new_ns

        for s, vi in enumerate(new_vs):
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

            if print_conv:
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

        new_vs = ()
        new_ns = ()
        rmaxv = numpy.zeros((neig,))

        for start, end in iterate_block_boundaries(neig, nsvec):
            wi = w[start:end]
            yi = y[:, start:end]

            blks = tuple(itertools.accumulate(ns))
            yis = numpy.split(yi, blks, axis=0)
            axi = sum(numpy.dot(avj, yij) for avj, yij in zip(avs, yis))
            bxi = sum(numpy.dot(bvj, yij) for bvj, yij in zip(bvs, yis))

            ri = axi - bxi * wi
            rmaxv[start:end] = numpy.amax(numpy.abs(ri), axis=0)
            precnd = numpy.reshape(wi[None, :] * bd[:, None] - ad[:, None],
                                   ri.shape)
            vi = ri / precnd
            for vj in vs + new_vs:
                vj = numpy.array(vj)
                vi -= numpy.linalg.multi_dot([vj, vj.T, vi])
            vi, s, _ = scipy.linalg.svd(vi, full_matrices=False,
                                        overwrite_a=True)
            tol = (max(vi.shape) * numpy.amax(numpy.abs(vi)) *
                   numpy.finfo(float).eps)
            vi = vi[:, s > tol]
            vi = dataset(vi) if disk else vi
            _, ni = vi.shape
            new_vs += (vi,)
            new_ns += (ni,)

        rdim = sum(ns)
        rdim_new = sum(new_ns)

        rmax = max(rmaxv)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        converged = rmax < rthresh

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:15.10f}, rmax={:3.1e}'
                      .format(j, wj, rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + rdim_new > nvec:
            if disk:
                consume(map(remove_dataset, new_vs))

            new_vs = ()

            for start, end in iterate_block_boundaries(neig, nsvec):
                ni = end-start
                wi = w[start:end]
                yi = y[:, start:end]

                blks = tuple(itertools.accumulate(ns))
                yis = numpy.split(yi, blks, axis=0)
                xi = sum(numpy.dot(vj, yij) for vj, yij in zip(vs, yis))
                xi = dataset(xi) if disk else xi
                new_vs += (xi,)

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

    if disk:
        consume(map(remove_dataset, new_vs))
        consume(map(remove_dataset, vs))
        consume(map(remove_dataset, avs))
        consume(map(remove_dataset, bvs))

    if not converged:
        warnings.warn("Did not converge! (rmax: {:7.1e})".format(rmax))

    x = empty_dataset((dim, neig))
    for start, end in iterate_block_boundaries(neig, nsvec):
        ni = end-start
        wi = w[start:end]
        yi = y[:, start:end]

        blks = tuple(itertools.accumulate(ns))
        yis = numpy.split(yi, blks, axis=0)
        xi = sum(numpy.dot(vj, yij) for vj, yij in zip(vs, yis))
        x[:, start:end] = xi

    return w, x, info
