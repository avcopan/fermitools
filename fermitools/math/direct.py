import sys
import numpy
import scipy
import itertools
import warnings
from .ix import cast


def adjoint(x):
    return numpy.conj(numpy.transpose(x))


def nslice(n, highest=False):
    return slice(None, n) if not highest else slice(None, -n-1, -1)


def eigenvector_guess(m, ad, bd=None, highest=False):
    bd = bd if bd is not None else numpy.ones_like(ad)

    n = len(ad)
    rows = numpy.argsort(ad/bd)[::-1] if highest else numpy.argsort(ad/bd)

    v = numpy.zeros((n, m))

    for row, col in zip(rows, range(m)):
        v[row, col] = 1.

    return v


def orth(x, compress=False):
    tol = (max(x.shape) * numpy.amax(numpy.abs(x)) * numpy.finfo(float).eps)
    x, s, _ = scipy.linalg.svd(x, full_matrices=False, overwrite_a=True)
    return x if not compress else x[:, s > tol]


def biorth(x, y):
    m, n = numpy.shape(x)
    tol = m * numpy.finfo(float).eps

    xs = ()
    ys = ()

    for i in range(n):
        x[:, i] -= numpy.linalg.multi_dot([x[:, :i], adjoint(y[:, :i]),
                                           x[:, i]])
        y[:, i] -= numpy.linalg.multi_dot([y[:, :i], adjoint(x[:, :i]),
                                           y[:, i]])
        norm = numpy.dot(numpy.conj(x[:, i]), y[:, i])
        x[:, i] = x[:, i]
        y[:, i] = y[:, i] / norm

        if norm > tol:
            xs += (x[:, i][:, None],)
            ys += (y[:, i][:, None],)

    x = numpy.hstack(xs)
    y = numpy.hstack(ys)
    return x, y
#
#
# def biorth(x, y):
#     tolx = (max(x.shape) * numpy.amax(numpy.abs(x)) * numpy.finfo(float).eps)
#     toly = (max(y.shape) * numpy.amax(numpy.abs(y)) * numpy.finfo(float).eps)
#     x, sx, _ = scipy.linalg.svd(x, full_matrices=False, overwrite_a=True)
#     y, sy, _ = scipy.linalg.svd(y, full_matrices=False, overwrite_a=True)
#     x = x[:, numpy.logical_or(sx > tolx, sy > toly)]
#     y = y[:, numpy.logical_or(sx > tolx, sy > toly)]
#     binorm = numpy.sqrt(numpy.sum(numpy.conj(x) * y, axis=0))
#     x = x / cast(binorm, 1, 2)
#     y = y / cast(binorm, 1, 2)
#     return x, y


def project_out(x, ys, yduals=None):
    yduals = yduals if yduals is not None else ys
    for y, ydual in zip(ys, yduals):
        x -= numpy.linalg.multi_dot([y, adjoint(ydual), x])
    return x


def tril_bmat(trilbls):
    mat, trilbls = numpy.bmat(trilbls[0]), trilbls[1:]
    for blrow in trilbls:
        row = numpy.bmat(blrow)
        col = numpy.transpose(numpy.zeros_like(numpy.bmat(blrow[:-1])))
        mat = numpy.hstack((mat, col))
        mat = numpy.vstack((mat, row))
    return mat


def triu_bmat(triubls):
    mat, triubls = numpy.bmat(triubls[0]), triubls[1:]
    for blcol in triubls:
        col = numpy.bmat(tuple(zip(blcol)))
        row = numpy.transpose(numpy.zeros_like(
            numpy.bmat(tuple(zip(blcol[:-1])))))
        mat = numpy.vstack((mat, row))
        mat = numpy.hstack((mat, col))
    return mat


def combine_triangles(tril, triu):
    n, _ = numpy.shape(tril)
    dt = tril.dtype
    mat = numpy.zeros((n, n), dtype=dt)
    mat[numpy.tril_indices(n)] = tril[numpy.tril_indices(n)]
    mat[numpy.triu_indices(n, +1)] = triu[numpy.triu_indices(n, +1)]
    return mat


def split_rdot(xs, y):
    splits = tuple(itertools.accumulate(numpy.shape(x)[-1] for x in xs))[:-1]
    ys = numpy.split(y, splits, axis=-2)
    return sum(numpy.dot(x, y) for x, y in zip(xs, ys))


def eigh(a, k, ad, b=None, bd=None, nconv=None, x0=None, nx0=None,
         highest=False, maxvecs=None, maxiter=100, tol=1e-5, print_conv=False):
    b = b if b is not None else (lambda x: x)
    bd = bd if bd is not None else numpy.ones_like(ad)
    nconv = nconv if nconv is not None else k
    assert 0 < nconv <= k
    x0 = x0 if x0 is not None else eigenvector_guess(m=nx0, ad=ad, bd=bd,
                                                     highest=highest)
    _, nx0 = numpy.shape(x0)
    maxvecs = maxvecs if maxvecs is not None else nx0 + 40 * k

    kslc = nslice(k, highest=highest)

    xs = ()
    axs = ()
    bxs = ()

    ap_trilbls = ()
    bp_trilbls = ()

    xi = x0

    for iteration in range(maxiter):
        xi = orth(project_out(xi, xs), compress=True)
        axi = a(xi)
        bxi = b(xi)
        xs += (xi,)
        axs += (axi,)
        bxs += (bxi,)
        ap_trilbls += (tuple(numpy.dot(numpy.transpose(xi), axj)
                             for axj in axs),)
        bp_trilbls += (tuple(numpy.dot(numpy.transpose(xi), bxj)
                             for bxj in bxs),)
        ap_tril = tril_bmat(ap_trilbls)
        bp_tril = tril_bmat(bp_trilbls)
        ap = combine_triangles(tril=ap_tril, triu=numpy.transpose(ap_tril))
        bp = combine_triangles(tril=bp_tril, triu=numpy.transpose(bp_tril))
        vals, vecs = scipy.linalg.eigh(a=ap, b=bp)
        w = vals[kslc]
        vp = vecs[:, kslc]

        v = split_rdot(xs, vp)
        av = split_rdot(axs, vp)
        bv = split_rdot(bxs, vp)

        r = av - bv * w
        xi = -r / (cast(ad, 0, 2) - cast(bd, 0, 2) * cast(w, 1, 2))

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:13.10f}, rmax={:3.1e}'
                      .format(j, wj, rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + k > maxvecs:
            xs = ()
            axs = ()
            bxs = ()

            ap_trilbls = ()
            bp_trilbls = ()
            xi = v


def eig(a, k, ad, nconv=None, x0=None, nx0=None, highest=False, maxvecs=None,
        maxiter=100, tol=1e-5, print_conv=False):
    nconv = nconv if nconv is not None else k
    assert 0 < nconv <= k
    x0 = x0 if x0 is not None else eigenvector_guess(m=nx0, ad=ad,
                                                     highest=highest)
    _, nx0 = numpy.shape(x0)
    maxvecs = maxvecs if maxvecs is not None else nx0 + 40 * k

    kslc = nslice(k, highest=highest)

    xs = ()
    axs = ()

    ap_trilbls = ()
    ap_triubls = ()

    xi = x0

    for iteration in range(maxiter):
        xi = orth(project_out(xi, xs), compress=True)
        axi = a(xi)
        xs += (xi,)
        axs += (axi,)
        ap_rowi = tuple(numpy.dot(numpy.transpose(xi), axj) for axj in axs)
        ap_coli = tuple(numpy.dot(numpy.transpose(xj), axi) for xj in xs)
        ap_trilbls += (ap_rowi,)
        ap_triubls += (ap_coli,)
        ap_tril = tril_bmat(ap_trilbls)
        ap_triu = triu_bmat(ap_triubls)
        ap = combine_triangles(tril=ap_tril, triu=ap_triu)
        vals, vecs = scipy.linalg.eig(a=ap)
        srt = numpy.argsort(vals)
        w = numpy.real(vals[srt[kslc]])
        vp = numpy.real(vecs[:, srt[kslc]])
        print('Norm of complex part:')
        print(scipy.linalg.norm(numpy.imag(vecs[:, srt[kslc]])))

        v = split_rdot(xs, vp)
        av = split_rdot(axs, vp)

        r = av - cast(w, 1, 2)
        xi = -r / (cast(ad, 0, 2) - cast(w, 1, 2))

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:13.10f}, rmax={:3.1e}'
                      .format(j, wj, rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + k > maxvecs:
            xs = ()
            axs = ()

            ap_trilbls = ()
            xi = v


def orth_against(a, against=None, tol=None):
    m, n = a.shape
    b = numpy.zeros((m, 0)) if against is None else against
    tol = max(m, n) * numpy.finfo(float).eps if tol is None else tol

    a_proj = a - numpy.linalg.multi_dot([b, b.T, a])
    a_orth, svals, _ = scipy.linalg.svd(a_proj,
                                        full_matrices=False,
                                        overwrite_a=True)
    nkeep = numpy.sum(svals > tol, dtype=int)
    return a_orth[:, :nkeep]


def eighg_simple(a, b, neig, ad, bd, nguess, niter=100, nvec=100, rthresh=1e-5,
                 print_conv=True, highest=False):
    guess = eigenvector_guess(m=nguess, ad=ad, bd=bd, highest=highest)
    dim, _ = guess.shape

    v1 = guess
    av = bv = v = numpy.zeros((dim, 0))

    slc = slice(None, neig) if not highest else slice(None, -neig-1, -1)

    for iteration in range(niter):
        v = numpy.concatenate((v, v1), axis=1)
        av = numpy.concatenate((av, a(v1)), axis=1)
        bv = numpy.concatenate((bv, b(v1)), axis=1)
        _, rdim = v.shape

        a_red = numpy.dot(v.T, av)
        b_red = numpy.dot(v.T, bv)

        vals, vecs = scipy.linalg.eigh(a=a_red, b=b_red)

        w = vals[slc]
        x_red = vecs[:, slc]

        x = numpy.dot(v, x_red)
        ax = numpy.dot(av, x_red)
        bx = numpy.dot(bv, x_red)

        r = ax - bx * w
        rmax = numpy.amax(numpy.abs(r))

        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        converged = rmax < rthresh

        if print_conv:
            print(info)
            # (TEMPORARY HACK -- DELETE THIS LATER)
            print(1/w)
            sys.stdout.flush()

        if converged:
            break

        denom = numpy.reshape(w[None, :] * bd[:, None] - ad[:, None], r.shape)
        vstep = r / denom
        v1 = orth_against(vstep, against=v)
        _, rdim1 = v1.shape

        if rdim + rdim1 > nvec:
            av = bv = v = numpy.zeros((dim, 0))
            v1 = x

    if not converged:
        warnings.warn("Did not converge! (rmax: {:7.1e})".format(rmax))

    return w, x, info


def eig_simple(a, b, neig, ad, bd, nguess, niter=100, nvec=100, rthresh=1e-5,
               print_conv=True, highest=False):
    guess = eigenvector_guess(m=nguess, ad=ad, bd=bd, highest=highest)
    dim, _ = guess.shape

    v1 = guess
    av = bv = v = numpy.zeros((dim, 0))

    slc = slice(None, neig) if not highest else slice(None, -neig-1, -1)

    for iteration in range(niter):
        v = numpy.concatenate((v, v1), axis=1)
        av = numpy.concatenate((av, a(v1)), axis=1)
        bv = numpy.concatenate((bv, b(v1)), axis=1)
        _, rdim = v.shape

        a_red = numpy.dot(v.T, av)
        b_red = numpy.dot(v.T, bv)

        vals, vecs = scipy.linalg.eig(a=a_red, b=b_red)

        srt = numpy.argsort(vals)
        w = numpy.real(vals[srt[slc]])
        x_red = numpy.real(vecs[:, srt[slc]])

        x = numpy.dot(v, x_red)
        ax = numpy.dot(av, x_red)
        bx = numpy.dot(bv, x_red)

        r = ax - bx * w
        rmax = numpy.amax(numpy.abs(r))

        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        converged = rmax < rthresh

        if print_conv:
            print(info)
            # (TEMPORARY HACK -- DELETE THIS LATER)
            print(numpy.sqrt(w))
            sys.stdout.flush()

        if converged:
            break

        denom = numpy.reshape(w[None, :] * bd[:, None] - ad[:, None], r.shape)
        vstep = r / denom
        v1 = orth_against(vstep, against=v)
        _, rdim1 = v1.shape

        if rdim + rdim1 > nvec:
            av = bv = v = numpy.zeros((dim, 0))
            v1 = x

    if not converged:
        warnings.warn("Did not converge! (rmax: {:7.1e})".format(rmax))

    return w, x, info
