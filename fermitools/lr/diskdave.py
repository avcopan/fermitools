import numpy
import warnings
import scipy.linalg

from functools import partial
from itertools import accumulate
from more_itertools import windowed
from more_itertools import consume

from .coredave import block_count, update_block_rep, block_dict_matrix
from .coredave import split_rdot
from ..math.ix import cast
from ..math.direct import standard_basis_vectors
from ..math.direct import project_out, orth

import os
import sys
import h5py
import tempfile


def eig(a, k, ad, b=None, bd=None, nconv=None, blsize=None, nguess=None,
        maxdim=None, maxiter=100, tol=1e-5, print_conv=False, printf=None,
        sym=False):
    s, k = numpy.sign(k), numpy.abs(k)
    nconv = nconv if nconv is not None else k
    blsize = blsize if blsize is not None else k
    nguess = nguess if nguess is not None else 2*k
    maxdim = maxdim if maxdim is not None else nguess + 40*k
    printf = printf if printf is not None else (lambda x: x)
    bd = bd if b is not None else numpy.ones_like(ad)
    prefix = tempfile.mkstemp()[1]

    f_ = partial(file_name, prefix)

    assert isinstance(bd, numpy.ndarray)
    assert nconv <= k <= nguess <= maxdim

    dim = len(ad)

    us = ()
    aus = ()
    bus = () if b is not None else None

    apd = {}
    bpd = {} if b is not None else None

    axes = (numpy.argsort(ad/bd)[:nguess] if s > 0 else
            numpy.argsort(ad/bd)[-nguess:])
    blocks = numpy.array_split(axes, block_count(nguess, blsize))
    ds = tuple(dataset(f_('d', i), data=standard_basis_vectors(dim, block))
               for i, block in enumerate(blocks))

    converged = False

    for iteration in range(maxiter):
        for di in ds:
            i = len(us)
            ui = orth(project_out(di, us))
            if numpy.size(ui) > 0:
                ui = dataset(f_('u', i), data=ui)
                aui = dataset(f_('au', i), data=a(ui))
                bui = (dataset(f_('bu', i), data=b(ui))
                       if b is not None else None)
                apd = update_block_rep(apd, ui, aui, us, aus, sym=sym)
                bpd = (update_block_rep(bpd, ui, bui, us, bus, sym=sym)
                       if b is not None else None)
                us += (ui,)
                aus += (aui,)
                bus = bus + (bui,) if b is not None else None
                if print_conv:
                    print("block {:d}".format(i))

        consume(map(remove_dataset, ds))

        nbl = len(us)
        ap = block_dict_matrix(apd, (nbl, nbl))
        bp = block_dict_matrix(bpd, (nbl, nbl)) if b is not None else None

        vals, vecs = (scipy.linalg.eigh(a=ap, b=bp) if sym else
                      scipy.linalg.eig(a=ap, b=bp))
        select = (numpy.argsort(vals)[:k] if s > 0 else
                  numpy.argsort(vals)[-k:])
        w = vals[select]
        vp = vecs[:, select]

        ws = numpy.array_split(w, block_count(k, blsize))
        vps = numpy.array_split(vp, block_count(k, blsize), axis=1)

        rmaxvs = ()
        imaxvs = ()

        vs = ()
        ds = ()

        for i, (wi, vpi) in enumerate(zip(ws, vps)):
            vi = split_rdot(us, vpi)
            avi = split_rdot(aus, vpi)
            bvi = split_rdot(bus, vpi) if b is not None else vi
            ri = avi - bvi * cast(wi, 1, 2)
            di = -ri / (cast(ad, 0, 2) - cast(bd, 0, 2) * cast(wi, 1, 2))
            rmaxvi = numpy.amax(numpy.abs(ri), axis=0)
            rmaxvs += (rmaxvi,)

            imaxvi = numpy.amax(numpy.abs(numpy.imag(vi)), axis=0)
            imaxvs += (imaxvi,)

            vi = dataset(f_('v', i), data=vi)
            di = dataset(f_('d', i), data=di)

            vs += (vi,)
            ds += (di,)

        rmaxv = numpy.concatenate(rmaxvs)
        imaxv = numpy.concatenate(imaxvs)
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
            if b is not None:
                consume(map(remove_dataset, bus))

            us = ()
            aus = ()
            bus = () if b is not None else None

            apd = {}
            bpd = {} if b is not None else None
            ds = vs
        else:
            consume(map(remove_dataset, vs))

    dtype = numpy.float64 if max(imaxv) < tol else numpy.complex128
    v = empty_dataset(f_('v'), shape=(dim, k), dtype=dtype)

    fill_col_slices(v, vs)

    consume(map(remove_dataset, vs))
    consume(map(remove_dataset, ds))
    consume(map(remove_dataset, us))
    consume(map(remove_dataset, aus))
    os.remove(prefix)

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return w, v, info


def file_name(prefix, label, number=None):
    return ('{:s}.{:s}'.format(prefix, label) if number is None else
            '{:s}.{:s}.{:d}'.format(prefix, label, number))


def empty_dataset(fname, shape, dtype=numpy.float64):
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', shape, dtype=dtype)


def dataset(fname, data):
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', data=data, dtype=data.dtype)


def remove_dataset(dataset):
    os.remove(dataset.file.filename)


def fill_col_slices(x, ys):
    stops = accumulate([0] + [numpy.shape(y)[1] for y in ys])
    for y, (start, stop) in zip(ys, windowed(stops, 2)):
        if not numpy.iscomplexobj(x):
            y = numpy.real(y)
        x[:, start:stop] = y
    return x
