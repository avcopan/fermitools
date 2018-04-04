import sys
import numpy
import scipy.linalg
import itertools
import warnings
from .ix import cast

import os
import h5py
import tempfile
from more_itertools import consume


def standard_basis_vectors(dim, axes):
    xs = list(axes)

    n = int(dim)
    m = len(xs)
    u = numpy.zeros((n, m))

    for i, j in zip(xs, range(m)):
        u[i, j] = 1.

    return u


def project_out(x, ys):
    for y in ys:
        x -= numpy.linalg.multi_dot([y, numpy.transpose(y), x])
    return x


def orth(x, compress=True):
    tol = (max(x.shape) * numpy.amax(numpy.abs(x)) * numpy.finfo(float).eps)
    if numpy.amax(numpy.abs(numpy.imag(x))) > tol:
        x = numpy.hstack([numpy.real(x), numpy.imag(x)])
    else:
        x = numpy.real(x)
    x, s, _ = scipy.linalg.svd(x, full_matrices=False, overwrite_a=True)
    return x if not compress else x[:, s > tol]


def update_block_rep(xi, axi, bldict, xs, axs, sym=False):
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
    ys = numpy.split(y, splits, axis=-2)
    return sum(numpy.dot(x, y) for x, y in zip(xs, ys))


def solve(a, b, ad, maxdim=None, maxiter=50, tol=1e-5, print_conv=False):
    dim = len(ad)
    ndim = numpy.ndim(b)
    maxdim = maxdim if maxdim is not None else dim

    guess = b / cast(ad, 0, ndim)

    au = u = numpy.zeros((dim, 0))

    converged = False

    ui = guess

    for iteration in range(maxiter):
        ui = orth(project_out(ui, (u,)), compress=True)
        aui = a(ui)

        u = numpy.concatenate((u, ui), axis=1)
        au = numpy.concatenate((au, aui), axis=1)

        ap = numpy.dot(numpy.transpose(u), au)
        bp = numpy.dot(numpy.transpose(u), b)
        xp = scipy.linalg.solve(ap, bp)

        x = numpy.dot(u, xp)
        ax = numpy.dot(au, xp)
        r = ax - b
        ui = -r / cast(ad, 0, ndim)

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
        rmax = max(rmaxv)
        converged = rmax < tol
        rdim, _ = numpy.shape(ap)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if print_conv:
            print(info)
            print("Residuals:")
            print(rmaxv)

        if converged:
            break

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return x, info


def eig_simple(a, k, ad, nconv=None, nguess=None, maxdim=None, maxiter=100,
               tol=1e-5, print_conv=False, printf=None):
    nconv = nconv if nconv is not None else k
    nguess = nguess if nguess is not None else 2*k
    maxdim = maxdim if maxdim is not None else nguess + 40*k
    printf = printf if printf is not None else (lambda x: x)

    dim = len(ad)
    axes = numpy.argsort(ad)[numpy.arange(nguess)]
    guess = standard_basis_vectors(dim=dim, axes=axes)

    au = u = numpy.zeros((dim, 0))

    converged = False

    ui = guess

    for iteration in range(maxiter):
        ui = orth(project_out(ui, (u,)), compress=True)
        aui = a(ui)

        u = numpy.concatenate((u, ui), axis=1)
        au = numpy.concatenate((au, aui), axis=1)

        ap = numpy.dot(numpy.transpose(u), au)

        vals, vecs = scipy.linalg.eig(a=ap)

        select = numpy.argsort(vals)[:k]
        w = vals[select]
        vp = vecs[:, select]

        v = numpy.dot(u, vp)
        av = numpy.dot(au, vp)

        r = av - v * cast(w, 1, 2)
        ui = -r / (cast(ad, 0, 2) - cast(w, 1, 2))

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
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
            au = u = numpy.zeros((dim, 0))
            ui = v

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return w, v, info


def eig(a, k, ad, nconv=None, nguess=None, maxdim=None, maxiter=100, tol=1e-5,
        print_conv=False, printf=None):
    nconv = nconv if nconv is not None else k
    nguess = nguess if nguess is not None else 2*k
    maxdim = maxdim if maxdim is not None else nguess + 40*k
    printf = printf if printf is not None else (lambda x: x)

    assert nconv <= k <= nguess <= maxdim

    dim = len(ad)
    axes = numpy.argsort(ad)[numpy.arange(nguess)]
    guess = standard_basis_vectors(dim=dim, axes=axes)

    us = ()
    aus = ()

    ap_bldict = {}

    converged = False

    ui = guess

    for iteration in range(maxiter):
        ui = orth(project_out(ui, us), compress=True)
        aui = a(ui)

        ap_bldict = update_block_rep(ui, aui, ap_bldict, us, aus)
        us += (ui,)
        aus += (aui,)

        nbl = len(us)
        ap = block_dict_matrix(ap_bldict, (nbl, nbl))

        vals, vecs = scipy.linalg.eig(a=ap)

        select = numpy.argsort(vals)[:k]
        w = numpy.real(vals[select])
        vp = numpy.real(vecs[:, select])
        vp_im_norm = scipy.linalg.norm(numpy.imag(vecs[:, select]))

        v = split_rdot(us, vp)
        av = split_rdot(aus, vp)

        r = av - v * cast(w, 1, 2)
        ui = -r / (cast(ad, 0, 2) - cast(w, 1, 2))

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if vp_im_norm > numpy.finfo(float).eps:
            warnings.warn("Discarding imaginary component with norm {:3.1e}."
                          .format(vp_im_norm))

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:13.10f}, rmax={:3.1e}'
                      .format(j, printf(wj), rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + k > maxdim:
            us = ()
            aus = ()
            ap_bldict = {}
            ui = v

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return w, v, info


def eigh(a, k, ad, b=None, bd=None, nconv=None, nguess=None, maxdim=None,
         maxiter=100, tol=1e-5, print_conv=False, printf=None):
    nconv = nconv if nconv is not None else abs(k)
    nguess = nguess if nguess is not None else 2*abs(k)
    maxdim = maxdim if maxdim is not None else nguess + 40*abs(k)
    printf = printf if printf is not None else (lambda x: x)
    b = b if b is not None else (lambda x: x)
    bd = bd if bd is not None else numpy.ones_like(ad)

    assert nconv <= abs(k) <= nguess <= maxdim

    dim = len(ad)
    axes = (numpy.argsort(ad/bd)[:nguess] if k > 0 else
            numpy.argsort(ad/bd)[-nguess:])
    guess = standard_basis_vectors(dim=dim, axes=axes)

    us = ()
    aus = ()
    bus = ()

    ap_bldict = {}
    bp_bldict = {}

    converged = False

    ui = guess

    for iteration in range(maxiter):
        ui = orth(project_out(ui, us), compress=True)
        aui = a(ui)
        bui = b(ui)

        ap_bldict = update_block_rep(ui, aui, ap_bldict, us, aus, sym=True)
        bp_bldict = update_block_rep(ui, bui, bp_bldict, us, bus, sym=True)
        us += (ui,)
        aus += (aui,)
        bus += (bui,)

        nbl = len(us)
        ap = block_dict_matrix(ap_bldict, (nbl, nbl))
        bp = block_dict_matrix(bp_bldict, (nbl, nbl))

        vals, vecs = scipy.linalg.eigh(a=ap, b=bp)
        select = numpy.argsort(vals)[:k] if k > 0 else numpy.argsort(vals)[k:]
        w = vals[select]
        vp = vecs[:, select]

        v = split_rdot(us, vp)
        av = split_rdot(aus, vp)
        bv = split_rdot(bus, vp)

        r = av - bv * cast(w, 1, 2)
        ui = -r / (cast(ad, 0, 2) - cast(bd, 0, 2) * cast(w, 1, 2))

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:13.10f}, rmax={:3.1e}'
                      .format(j, printf(wj), rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + abs(k) > maxdim:
            us = ()
            aus = ()
            bus = ()
            ap_bldict = {}
            bp_bldict = {}
            ui = v

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return w, v, info


# Disk algorithms
def file_name(prefix, label, number):
    return '{:s}.{:s}.{:d}'.format(prefix, label, number)


def empty_dataset(fname, shape):
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', shape)


def dataset(fname, data):
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', data=data)


def remove_dataset(dataset):
    os.remove(dataset.file.filename)


def eig_disk(a, k, ad, nconv=None, blsize=None, nguess=None, maxdim=None,
             maxiter=100, tol=1e-5, print_conv=False, printf=None):
    nconv = nconv if nconv is not None else k
    blsize = blsize if blsize is not None else k
    nguess = nguess if nguess is not None else 2*k
    maxdim = maxdim if maxdim is not None else nguess + 40*k
    printf = printf if printf is not None else (lambda x: x)

    dim = len(ad)

    us = ()
    aus = ()

    ap_bldict = {}

    converged = False

    _, prf = tempfile.mkstemp()

    v = empty_dataset(fname=prf, shape=(dim, k))

    axes = numpy.argsort(ad)[numpy.arange(nguess)]
    splits = numpy.arange(blsize, nguess, blsize)
    new_us = tuple(dataset(fname=file_name(prf, 'new_u', i),
                           data=standard_basis_vectors(dim=dim, axes=ax))
                   for i, ax in enumerate(numpy.split(axes, splits)))

    for iteration in range(maxiter):

        for new_ui in new_us:
            i = len(us)
            ui = orth(project_out(new_ui, us), compress=True)
            if numpy.size(ui) > 0:
                ui = dataset(fname=file_name(prf, 'u', i), data=ui)
                aui = dataset(fname=file_name(prf, 'au', i), data=a(ui))
                ap_bldict = update_block_rep(ui, aui, ap_bldict, us, aus)
                us += (ui,)
                aus += (aui,)
            if print_conv:
                print("block {:d}".format(i))

        consume(map(remove_dataset, new_us))

        nbl = len(us)
        ap = block_dict_matrix(ap_bldict, (nbl, nbl))

        vals, vecs = scipy.linalg.eig(a=ap)

        select = numpy.argsort(vals)[:k]
        w = numpy.real(vals[select])
        vp = numpy.real(vecs[:, select])
        vp_im_norm = scipy.linalg.norm(numpy.imag(vecs[:, select]))

        ws = numpy.split(w, numpy.arange(blsize, k, blsize))
        vps = numpy.split(vp, numpy.arange(blsize, k, blsize), axis=1)

        rmaxvs = ()

        vs = ()
        new_us = ()

        for i, (wi, vpi) in enumerate(zip(ws, vps)):
            vi = split_rdot(us, vpi)
            avi = split_rdot(aus, vpi)
            ri = avi - vi * cast(wi, 1, 2)
            new_ui = -ri / (cast(ad, 0, 2) - cast(wi, 1, 2))
            rmaxvi = numpy.amax(numpy.abs(ri), axis=0)
            rmaxvs += (rmaxvi,)

            vi = dataset(fname=file_name(prf, 'v', i), data=vi)
            new_ui = dataset(fname=file_name(prf, 'new_u', i), data=new_ui)

            vs += (vi,)
            new_us += (new_ui,)

        rmaxv = numpy.concatenate(rmaxvs)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if vp_im_norm > numpy.finfo(float).eps:
            warnings.warn("Discarding imaginary component with norm {:3.1e}."
                          .format(vp_im_norm))

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:13.10f}, rmax={:3.1e}'
                      .format(j, printf(wj), rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + k > maxdim:
            consume(map(remove_dataset, us))
            consume(map(remove_dataset, aus))

            us = ()
            aus = ()

            ap_bldict = {}
            new_us = vs
        else:
            consume(map(remove_dataset, vs))

    consume(map(remove_dataset, new_us))
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
