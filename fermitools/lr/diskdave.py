import numpy
import itertools
import warnings
import scipy.linalg

from functools import partial
from more_itertools import consume

from ..math.ix import cast
from ..math.direct import standard_basis_vectors
from ..math.direct import project_out, orth

import os
import sys
import h5py
import tempfile


def eig(a, k, ad, nconv=None, blsize=None, nguess=None, maxdim=None,
        maxiter=100, tol=1e-5, print_conv=False, printf=None):
    nconv = nconv if nconv is not None else k
    blsize = blsize if blsize is not None else k
    nguess = nguess if nguess is not None else 2*k
    maxdim = maxdim if maxdim is not None else nguess + 40*k
    printf = printf if printf is not None else (lambda x: x)
    prefix = tempfile.mkstemp()[1]

    assert nconv <= k <= nguess <= maxdim

    dim = len(ad)

    us = ()
    aus = ()

    apd = {}

    converged = False

    v = empty_dataset(fname=file_name(prefix, 'v'), shape=(dim, k))
    axes = numpy.array_split(
            numpy.argsort(ad)[:nguess], block_count(nguess, blsize))
    guesses = map(partial(standard_basis_vectors, dim), axes)
    ds = tuple(dataset(file_name(prefix, 'd', i), data=guess)
               for i, guess in enumerate(guesses))

    for iteration in range(maxiter):
        for di in ds:
            i = len(us)
            ui = orth(project_out(di, us))
            if numpy.size(ui) > 0:
                ui = dataset(file_name(prefix, 'u', i), data=ui)
                aui = dataset(file_name(prefix, 'au', i), data=a(ui))
                apd = update_block_rep(apd, ui, aui, us, aus)
                us += (ui,)
                aus += (aui,)
            if print_conv:
                print("block {:d}".format(i))

        consume(map(remove_dataset, ds))

        nbl = len(us)
        ap = block_dict_matrix(apd, (nbl, nbl))

        vals, vecs = scipy.linalg.eig(a=ap)

        select = numpy.argsort(vals)[:k]
        w = numpy.real(vals[select])
        vp = numpy.real(vecs[:, select])

        ws = numpy.array_split(w, block_count(k, blsize))
        vps = numpy.array_split(vp, block_count(k, blsize), axis=1)

        rmaxvs = ()

        vs = ()
        ds = ()

        for i, (wi, vpi) in enumerate(zip(ws, vps)):
            vi = split_rdot(us, vpi)
            avi = split_rdot(aus, vpi)
            ri = avi - vi * cast(wi, 1, 2)
            di = -ri / (cast(ad, 0, 2) - cast(wi, 1, 2))
            rmaxvi = numpy.amax(numpy.abs(ri), axis=0)
            rmaxvs += (rmaxvi,)

            vi = dataset(file_name(prefix, 'v', i), data=vi)
            di = dataset(file_name(prefix, 'd', i), data=di)

            vs += (vi,)
            ds += (di,)

        rmaxv = numpy.concatenate(rmaxvs)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                wj = printf(wj)
                print('  eigenvalue {:d}: {:13.10f}, im={:3.1e} rmax={:3.1e}'
                      .format(j, numpy.real(wj), numpy.imag(wj), rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + k > maxdim:
            consume(map(remove_dataset, us))
            consume(map(remove_dataset, aus))

            us = ()
            aus = ()

            apd = {}
            ds = vs
        else:
            consume(map(remove_dataset, vs))

    consume(map(remove_dataset, ds))
    consume(map(remove_dataset, us))
    consume(map(remove_dataset, aus))

    offset = 0
    for vi in vs:
        _, ki = numpy.shape(vi)
        v[:, offset:offset+ki] = vi
        offset += ki

    consume(map(remove_dataset, vs))

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return w, v, info


def file_name(prefix, label, number=None):
    return ('{:s}.{:s}'.format(prefix, label) if number is None else
            '{:s}.{:s}.{:d}'.format(prefix, label, number))


def empty_dataset(fname, shape):
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', shape)


def dataset(fname, data):
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', data=data)


def remove_dataset(dataset):
    os.remove(dataset.file.filename)


def block_count(dim, blsize):
    return -(-dim // blsize)


def update_block_rep(bldict, xi, axi, xs, axs, sym=False):
    i = len(xs)
    bldict[i, i] = numpy.dot(numpy.transpose(xi), axi)

    for j, (xj, axj) in enumerate(zip(xs, axs)):
        bldict[i, j] = numpy.dot(numpy.transpose(xi), axj)
        bldict[j, i] = (numpy.transpose(bldict[i, j]) if sym else
                        numpy.dot(numpy.transpose(xj), axi))

    return bldict


def block_dict_matrix(bldict, blshape):
    n, m = blshape
    return numpy.bmat([[bldict[i, j] for j in range(m)] for i in range(n)])


def split_rdot(xs, y):
    splits = tuple(itertools.accumulate(numpy.shape(x)[-1] for x in xs))[:-1]
    ys = numpy.split(y, splits, axis=0)
    return sum(numpy.dot(x, y) for x, y in zip(xs, ys))
