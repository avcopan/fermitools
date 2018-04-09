import sys
import numpy
import scipy.linalg
import itertools
import warnings
from .ix import cast


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


def orth(x):
    tol = (max(x.shape) * numpy.amax(numpy.abs(x)) * numpy.finfo(float).eps)
    if numpy.amax(numpy.abs(numpy.imag(x))) > tol:
        x = numpy.hstack([numpy.real(x), numpy.imag(x)])
    else:
        x = numpy.real(x)
    x, s, _ = scipy.linalg.svd(x, full_matrices=False, overwrite_a=True)
    return x[:, s > tol]


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
        ui = orth(project_out(ui, (u,)))
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


def eig(a, k, ad, nconv=None, nguess=None, maxdim=None, maxiter=100,
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
        ui = orth(project_out(ui, (u,)))
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

    bu = au = u = numpy.zeros((dim, 0))

    converged = False

    ui = guess

    for iteration in range(maxiter):
        ui = orth(project_out(ui, (u,)))
        aui = a(ui)
        bui = b(ui)

        u = numpy.concatenate((u, ui), axis=1)
        au = numpy.concatenate((au, aui), axis=1)
        bu = numpy.concatenate((bu, bui), axis=1)

        ap = numpy.dot(numpy.transpose(u), au)
        bp = numpy.dot(numpy.transpose(u), bu)

        vals, vecs = scipy.linalg.eigh(a=ap, b=bp)

        select = numpy.argsort(vals)[:k] if k > 0 else numpy.argsort(vals)[k:]
        w = vals[select]
        vp = vecs[:, select]

        v = numpy.dot(u, vp)
        av = numpy.dot(au, vp)
        bv = numpy.dot(bu, vp)

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
                wj = printf(wj)
                print('  eigenvalue {:d}: {:13.10f}, im={:3.1e} rmax={:3.1e}'
                      .format(j, numpy.real(wj), numpy.imag(wj), rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + abs(k) > maxdim:
            bu = au = u = numpy.zeros((dim, 0))
            ui = v

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return w, v, info
