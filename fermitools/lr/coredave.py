import numpy
import warnings
import scipy.linalg
from itertools import accumulate
from ..math.ix import cast
from ..math.direct import standard_basis_vectors
from ..math.direct import project_out, orth

import sys


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
    ds = tuple(standard_basis_vectors(dim, axes=block) for block in blocks)

    converged = False

    for iteration in range(maxiter):
        for di in ds:
            i = len(us)
            ui = orth(project_out(di, us))
            if numpy.size(ui) > 0:
                aui = a(ui)
                bui = b(ui) if b is not None else None
                apd = update_block_rep(apd, ui, aui, us, aus, sym=sym)
                bpd = (update_block_rep(bpd, ui, bui, us, bus, sym=sym)
                       if b is not None else None)
                us += (ui,)
                aus += (aui,)
                bus = bus + (bui,) if b is not None else None
                if print_conv:
                    print("block {:d}".format(i))
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
            us = ()
            aus = ()
            bus = () if b is not None else None

            apd = {}
            bpd = {} if b is not None else None
            ds = vs

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    v = numpy.concatenate(vs, axis=1)

    return w, v, info


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
    stops = tuple(accumulate(numpy.shape(x)[1] for x in xs))
    ys = numpy.split(y, stops[:-1], axis=0)
    z = 0.
    for x, y in zip(xs, ys):
        xtype = x.dtype if hasattr(x, 'dtype') else numpy.float64
        ytype = y.dtype if hasattr(y, 'dtype') else numpy.float64
        x = numpy.array(x, dtype=xtype)
        y = numpy.array(y, dtype=ytype)
        z += numpy.dot(x, y)
    return z
