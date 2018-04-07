import numpy
import scipy

import sys
import time
import h5py
import tempfile

from toolz import functoolz
from .linmap import eye, negative, add, subtract, block_diag, bmat
from ..math import cast
from ..math import einsum
from ..math import transform
from ..math import diagonal_indices as dix
from ..math import raveler, unraveler
from ..math.asym import megaraveler, megaunraveler
from ..math.asym import antisymmetrizer_product as asm
from ..math.spinorb import transform_onebody, transform_twobody

from ..math.direct import eigh, solve, eig

from ..oo.odc12 import fock_xy
from ..oo.odc12 import fancy_property
from ..oo.odc12 import onebody_density


def solve_static_response(h_ao, p_ao, r_ao, co, cv, t2, maxdim=None,
                          maxiter=20, rthresh=1e-5, print_conv=False):
    a, b, ad = build_hessian_blocks(h_ao, r_ao, co, cv, t2)
    s, sd = build_metric_blocks(t2)
    pg = build_property_gradient_blocks(p_ao, co, cv, t2)

    print("Second-order (static) properties:")
    e = add(a, b)
    v = -2*pg
    r, info = solve(a=e, b=v, ad=ad, maxdim=maxdim, tol=rthresh,
                    print_conv=True)
    alpha = numpy.dot(r.T, pg)
    print(alpha.round(12))
    return alpha


def solve_spectrum(h_ao, r_ao, co, cv, t2, nroot=1, nconv=None, nguess=None,
                   maxdim=None, maxiter=100, rthresh=1e-5, print_conv=False,
                   disk=False, blsize=None, p_ao=None):
    a, b, ad = build_hessian_blocks(h_ao, r_ao, co, cv, t2)
    s, sd = build_metric_blocks(t2)

    e = bmat([[a, b], [b, a]], 2)
    m = block_diag((s, negative(s)), (len(ad),))
    ed = numpy.concatenate((+ad, +ad))
    md = numpy.concatenate((+sd, -sd))

    tm = time.time()
    w_inv, z, info = eigh(
            a=m, k=-nroot, ad=md, b=e, bd=ed, nconv=nconv, nguess=nguess,
            maxdim=maxdim, maxiter=maxiter, tol=rthresh, print_conv=print_conv,
            printf=numpy.reciprocal)
    w = numpy.reciprocal(w_inv)
    x, y = numpy.split(z, 2)
    print("\nODC-12 excitation energies (in a.u.):")
    print(w.reshape(-1, 1))
    print("\nODC-12 excitation energies (in eV):")
    print(w.reshape(-1, 1)*27.2114)
    print('\nODC-12 linear response total time: {:8.1f}s'
          .format(time.time() - tm))
    sys.stdout.flush()

    if p_ao is not None:
        pg = build_property_gradient_blocks(p_ao, co, cv, t2)
        norms = numpy.diag(numpy.dot(x.T, s(x)) - numpy.dot(y.T, s(y)))
        t = numpy.dot(x.T, pg) + numpy.dot(y.T, pg)
        mu_trans = t * t / norms[:, None]

        print("\nODC-12 transition dipoles (a.u.):")
        print(mu_trans.round(12))
        print("\nODC-12 norm of transition dipoles (a.u.):")
        print(numpy.sqrt(numpy.diag(numpy.dot(mu_trans, mu_trans.T))
              .reshape(-1, 1)).round(12))
        sys.stdout.flush()

    return w, (x, y), info


def solve_spectrum2(h_ao, r_ao, co, cv, t2, nroot=1, nconv=None, nguess=None,
                    maxdim=None, maxiter=100, rthresh=1e-5, print_conv=False,
                    disk=False, blsize=None, p_ao=None):
    a, b, ad = build_hessian_blocks(h_ao, r_ao, co, cv, t2)
    s, _ = build_metric_blocks(t2)
    si, _ = build_metric_function_blocks(t2, f=numpy.reciprocal)

    h_plus = functoolz.compose(si, add(a, b))
    h_minus = functoolz.compose(si, subtract(a, b))

    h_bar = functoolz.compose(h_minus, h_plus)
    hd = ad * ad

    tm = time.time()
    w2, c_plus, info = eig(
            a=h_bar, k=nroot, ad=hd, nconv=nconv, nguess=nguess, maxdim=maxdim,
            maxiter=maxiter, tol=rthresh, print_conv=print_conv,
            printf=numpy.sqrt)

    w = numpy.real(numpy.sqrt(w2))
    x = (c_plus + h_plus(c_plus) / cast(w, 1, 2)) / 2.
    y = (c_plus - h_plus(c_plus) / cast(w, 1, 2)) / 2.

    print("\nODC-12 excitation energies (in a.u.):")
    print(w.reshape(-1, 1))
    print("\nODC-12 excitation energies (in eV):")
    print(w.reshape(-1, 1)*27.2114)
    print('\nODC-12 linear response total time: {:8.1f}s'
          .format(time.time() - tm))
    sys.stdout.flush()

    if p_ao is not None:
        pg = build_property_gradient_blocks(p_ao, co, cv, t2)
        norms = numpy.diag(numpy.dot(x.T, s(x)) - numpy.dot(y.T, s(y)))
        t = numpy.dot(x.T, pg) + numpy.dot(y.T, pg)
        mu_trans = t * t / norms[:, None]

        print("\nODC-12 transition dipoles (a.u.):")
        print(mu_trans.round(12))
        print("\nODC-12 norm of transition dipoles (a.u.):")
        print(numpy.sqrt(numpy.diag(numpy.dot(mu_trans, mu_trans.T))
              .reshape(-1, 1)).round(12))
        sys.stdout.flush()

    return w, (x, y), info


def solve_spectrum3(h_ao, r_ao, co, cv, t2, nroot=1, nconv=None, nguess=None,
                    maxdim=None, maxiter=100, rthresh=1e-5, print_conv=False,
                    disk=False, blsize=None, p_ao=None):
    a, b, ad = build_hessian_blocks(h_ao, r_ao, co, cv, t2)
    s, _ = build_metric_blocks(t2)
    f = functoolz.compose(numpy.reciprocal, numpy.sqrt)
    sir, _ = build_metric_function_blocks(t2, f=f)

    h_plus = functoolz.compose(sir, add(a, b), sir)
    h_minus = functoolz.compose(sir, subtract(a, b), sir)

    h_bar = functoolz.compose(h_minus, h_plus)
    hd = ad * ad

    tm = time.time()
    w2, c_plus, info = eig(
            a=h_bar, k=nroot, ad=hd, nconv=nconv, nguess=nguess, maxdim=maxdim,
            maxiter=maxiter, tol=rthresh, print_conv=print_conv,
            printf=numpy.sqrt)

    w = numpy.real(numpy.sqrt(w2))

    print("\nODC-12 excitation energies (in a.u.):")
    print(w.reshape(-1, 1))
    print("\nODC-12 excitation energies (in eV):")
    print(w.reshape(-1, 1)*27.2114)
    print('\nODC-12 linear response total time: {:8.1f}s'
          .format(time.time() - tm))
    sys.stdout.flush()

    x = (sir(c_plus) + sir(h_plus(c_plus)) / cast(w, 1, 2)) / 2.
    y = (sir(c_plus) - sir(h_plus(c_plus)) / cast(w, 1, 2)) / 2.

    if p_ao is not None:
        pg = build_property_gradient_blocks(p_ao, co, cv, t2)
        norms = numpy.diag(numpy.dot(x.T, s(x)) - numpy.dot(y.T, s(y)))
        t = numpy.dot(x.T, pg) + numpy.dot(y.T, pg)
        mu_trans = t * t / norms[:, None]

        print("\nODC-12 transition dipoles (a.u.):")
        print(mu_trans.round(12))
        print("\nODC-12 norm of transition dipoles (a.u.):")
        print(numpy.sqrt(numpy.diag(numpy.dot(mu_trans, mu_trans.T))
              .reshape(-1, 1)).round(12))
        sys.stdout.flush()

    return w, (x, y), info


# Helper functions for solver
def count_excitations(no, nv):
    n1 = no * nv
    n2 = no * (no - 1) * nv * (nv - 1) // 4
    return n1, n2


def build_ravelers(no, nv):
    r1 = raveler({0: (0, 1)})
    u1 = unraveler({0: {0: no, 1: nv}})
    r2 = megaraveler({0: ((0, 1), (2, 3))})
    u2 = megaunraveler({0: {(0, 1): no, (2, 3): nv}})
    return r1, r2, u1, u2


def build_hessian_blocks(h_ao, r_ao, co, cv, t2):
    hoo = transform_onebody(h_ao, (co, co))
    hov = transform_onebody(h_ao, (co, cv))
    hvv = transform_onebody(h_ao, (cv, cv))
    goooo = transform_twobody(r_ao, (co, co, co, co))
    gooov = transform_twobody(r_ao, (co, co, co, cv))
    goovv = transform_twobody(r_ao, (co, co, cv, cv))
    govov = transform_twobody(r_ao, (co, cv, co, cv))
    govvv = transform_twobody(r_ao, (co, cv, cv, cv))
    gvvvv = transform_twobody(r_ao, (cv, cv, cv, cv))
    m1oo, m1vv = onebody_density(t2)
    foo = fock_xy(hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fock_xy(hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fock_xy(hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)

    ad1u = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2u = twobody_hessian_zeroth_order_diagonal(foo, fvv, t2)
    a11u, b11u = onebody_hessian(foo, fvv, goooo, goovv, govov, gvvvv, t2)
    a12u, b12u, a21u, b21u = mixed_hessian(fov, gooov, govvv, t2)
    a22u, b22u = twobody_hessian(foo, fvv, goooo, govov, gvvvv, t2)

    no, _, nv, _ = t2.shape
    r1, r2, u1, u2 = build_ravelers(no, nv)
    ad1 = r1(ad1u)
    ad2 = r2(ad2u)
    a11 = functoolz.compose(r1, a11u, u1)
    b11 = functoolz.compose(r1, b11u, u1)
    a12 = functoolz.compose(r1, a12u, u2)
    b12 = functoolz.compose(r1, b12u, u2)
    a21 = functoolz.compose(r2, a21u, u1)
    b21 = functoolz.compose(r2, b21u, u1)
    a22 = functoolz.compose(r2, a22u, u2)
    b22 = functoolz.compose(r2, b22u, u2)

    n1, n2 = count_excitations(no, nv)
    a = bmat([[a11, a12], [a21, a22]], (n1,))
    b = bmat([[b11, b12], [b21, b22]], (n1,))

    ad = numpy.concatenate((ad1, ad2), axis=0)

    return a, b, ad


def build_metric_blocks(t2):
    no, _, nv, _ = t2.shape
    r1, _, u1, _ = build_ravelers(no, nv)
    s11u = onebody_metric(t2)
    s11 = functoolz.compose(r1, s11u, u1)

    n1, n2 = count_excitations(no, nv)
    s = block_diag((s11, eye), (n1,))
    sd = numpy.ones(n1+n2)
    return s, sd


def build_metric_function_blocks(t2, f):
    no, _, nv, _ = t2.shape
    r1, _, u1, _ = build_ravelers(no, nv)
    sf11u = onebody_metric_function(t2, f=f)
    sf11 = functoolz.compose(r1, sf11u, u1)

    n1, n2 = count_excitations(no, nv)
    sf = block_diag((sf11, eye), (n1,))
    sfd = numpy.ones(n1+n2)
    return sf, sfd


def build_property_gradient_blocks(p_ao, co, cv, t2):
    poo = transform_onebody(p_ao, (co, co))
    pov = transform_onebody(p_ao, (co, cv))
    pvv = transform_onebody(p_ao, (cv, cv))
    m1oo, m1vv = onebody_density(t2)
    fpoo = fancy_property(poo, m1oo)
    fpvv = fancy_property(pvv, m1vv)

    pg1u = onebody_property_gradient(pov, m1oo, m1vv)
    pg2u = twobody_property_gradient(fpoo, -fpvv, t2)

    no, _, nv, _ = t2.shape
    r1, r2, _, _ = build_ravelers(no, nv)
    pg1 = r1(pg1u)
    pg2 = r2(pg2u)
    pg = numpy.concatenate((pg1, pg2), axis=0)
    return pg


# The LR-ODC-12 equations:
def onebody_hessian_zeroth_order_diagonal(foo, fvv):
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    return - cast(eo, 0, 2) + cast(ev, 1, 2)


def twobody_hessian_zeroth_order_diagonal(foo, fvv, t2):
    m1oo, m1vv = onebody_density(t2)
    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)
    efo = numpy.diagonal(ffoo)
    efv = numpy.diagonal(ffvv)
    return (- cast(efo, 0, 4) - cast(efo, 1, 4)
            - cast(efv, 2, 4) - cast(efv, 3, 4))


def onebody_property_gradient(pov, m1oo, m1vv):
    return (
        + einsum('im,...ma->ia...', m1oo, pov)
        - einsum('...ie,ea->ia...', pov, m1vv))


def twobody_property_gradient(poo, pvv, t2):
    return (
        + asm('2/3')(
              einsum('...ac,ijcb->ijab...', pvv, t2))
        - asm('0/1')(
              einsum('...ik,kjab->ijab...', poo, t2)))


def onebody_hessian(foo, fvv, goooo, goovv, govov, gvvvv, t2):
    m1oo, m1vv = onebody_density(t2)
    fcoo = (numpy.dot(foo, m1oo)
            + 1./2 * einsum('imef,jmef->ij', goovv, t2)
            + 1./4 * einsum('imno,jmcd,nocd->ij', goooo, t2, t2)
            - einsum('iemf,mkec,jkfc->ij', govov, t2, t2))
    fcvv = (numpy.dot(fvv, m1vv)
            + 1./2 * einsum('mnae,mnbe->ab', goovv, t2)
            + 1./4 * einsum('aefg,klbe,klfg->ab', gvvvv, t2, t2)
            - einsum('nema,mkec,nkbc->ab', govov, t2, t2))
    fsoo = (fcoo + numpy.transpose(fcoo)) / 2.
    fsvv = (fcvv + numpy.transpose(fcvv)) / 2.

    def _a11(r1):
        a11 = (
            - einsum('ab,ib...->ia...', fsvv, r1)
            - einsum('ij,ja...->ia...', fsoo, r1)
            + einsum('ij,ab,jb...->ia...', foo, m1vv, r1)
            + einsum('ab,ij,jb...->ia...', fvv, m1oo, r1)
            - einsum('manb,mj,in,jb...->ia...', govov, m1oo, m1oo, r1)
            - einsum('iejf,af,eb,jb...->ia...', govov, m1vv, m1vv, r1)
            + einsum('jame,im,be,jb...->ia...', govov, m1oo, m1vv, r1)
            + einsum('ibme,jm,ae,jb...->ia...', govov, m1oo, m1vv, r1)
            - einsum('minj,nkac,mkbc,jb...->ia...', goooo, t2, t2, r1)
            + 1./2 * einsum('manb,micd,njcd,jb...->ia...', govov, t2, t2, r1)
            + 1./2 * einsum('iejf,klae,klbf,jb...->ia...', govov, t2, t2, r1)
            - einsum('aebf,jkec,ikfc,jb...->ia...', gvvvv, t2, t2, r1)
            - einsum('ibme,mkac,jkec,jb...->ia...', govov, t2, t2, r1)
            - einsum('jame,mkbc,ikec,jb...->ia...', govov, t2, t2, r1))
        return a11

    def _b11(r1):
        b11 = (
            + einsum('jema,imbe,jb...->ia...', govov, t2, r1)
            + einsum('iemb,jmae,jb...->ia...', govov, t2, r1)
            + 1./2 * einsum('ijmn,mnab,jb...->ia...', goooo, t2, r1)
            + 1./2 * einsum('efab,ijef,jb...->ia...', gvvvv, t2, r1)
            + einsum('imbe,jm,ea,jb...->ia...', goovv, m1oo, m1vv, r1)
            + einsum('jmae,im,eb,jb...->ia...', goovv, m1oo, m1vv, r1)
            + einsum('mnab,im,jn,jb...->ia...', goovv, m1oo, m1oo, r1)
            + einsum('ijef,ea,fb,jb...->ia...', goovv, m1vv, m1vv, r1)
            + 1./4 * einsum('mnab,ijcd,mncd,jb...->ia...', goovv, t2, t2, r1)
            - einsum('imbe,mkec,jkac,jb...->ia...', goovv, t2, t2, r1)
            - einsum('jmae,mkec,ikbc,jb...->ia...', goovv, t2, t2, r1)
            + 1./4 * einsum('ijef,klef,klab,jb...->ia...', goovv, t2, t2, r1))
        return b11

    return _a11, _b11


def mixed_hessian(fov, gooov, govvv, t2):
    m1oo, m1vv = onebody_density(t2)
    mo, uo = scipy.linalg.eigh(m1oo)
    mv, uv = scipy.linalg.eigh(m1vv)
    n1oo = cast(mo, 2, 4) + cast(mo, 3, 4) - 1
    n1vv = cast(mv, 2, 4) + cast(mv, 3, 4) - 1
    io = numpy.eye(*uo.shape)
    iv = numpy.eye(*uv.shape)
    ioo = numpy.ascontiguousarray(
           + einsum('ik,la->iakl', io, fov)
           - einsum('mlka,im->iakl', gooov, m1oo)
           + einsum('ilke,ae->iakl', gooov, m1vv))
    ivv = numpy.ascontiguousarray(
           - einsum('ac,id->iadc', iv, fov)
           + einsum('mcad,im->iadc', govvv, m1oo)
           - einsum('iced,ae->iadc', govvv, m1vv))
    tfioo = transform(ioo, (uo, uo)) / n1oo
    tfivv = transform(ivv, (uv, uv)) / n1vv
    uot = numpy.ascontiguousarray(numpy.transpose(uo))
    uvt = numpy.ascontiguousarray(numpy.transpose(uv))
    fioo = transform(tfioo, (uot, uot))
    fivv = transform(tfivv, (uvt, uvt))

    def _a12(r2):
        a12 = (
            - 1./2 * einsum('lacd,ilcd...->ia...', govvv, r2)
            - 1./2 * einsum('klid,klad...->ia...', gooov, r2)
            - 1./2 * einsum('iakm,mlcd,klcd...->ia...', fioo, t2, r2)
            - 1./2 * einsum('iaec,kled,klcd...->ia...', fivv, t2, r2)
            - einsum('mcae,mled,ilcd...->ia...', govvv, t2, r2)
            - einsum('imke,mled,klad...->ia...', gooov, t2, r2)
            - 1./4 * einsum('mnla,mncd,ilcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('idef,klef,klad...->ia...', govvv, t2, r2))
        return a12

    def _b12(r2):
        b12 = (
            - 1./2 * einsum('iamk,mlcd,klcd...->ia...', fioo, t2, r2)
            - 1./2 * einsum('iace,kled,klcd...->ia...', fivv, t2, r2)
            - einsum('lead,kice,klcd...->ia...', govvv, t2, r2)
            - einsum('ilmd,kmca,klcd...->ia...', gooov, t2, r2)
            + 1./4 * einsum('klma,micd,klcd...->ia...', gooov, t2, r2)
            + 1./4 * einsum('iecd,klea,klcd...->ia...', govvv, t2, r2))
        return b12

    def _a21(r1):
        a21 = asm('0/1|2/3')(
            - 1./2 * einsum('jcab,ic...->ijab...', govvv, r1)
            - 1./2 * einsum('ijkb,ka...->ijab...', gooov, r1)
            - 1./2 * einsum('kcim,mjab,kc...->ijab...', fioo, t2, r1)
            - 1./2 * einsum('kcea,ijeb,kc...->ijab...', fivv, t2, r1)
            - einsum('mace,mjeb,ic...->ijab...', govvv, t2, r1)
            - einsum('kmie,mjeb,ka...->ijab...', gooov, t2, r1)
            - 1./4 * einsum('mnjc,mnab,ic...->ijab...', gooov, t2, r1)
            - 1./4 * einsum('kbef,ijef,ka...->ijab...', govvv, t2, r1))
        return a21

    def _b21(r1):
        b21 = asm('0/1|2/3')(
            - 1./2 * einsum('kcmi,mjab,kc...->ijab...', fioo, t2, r1)
            - 1./2 * einsum('kcae,ijeb,kc...->ijab...', fivv, t2, r1)
            - einsum('jecb,ikae,kc...->ijab...', govvv, t2, r1)
            - einsum('kjmb,imac,kc...->ijab...', gooov, t2, r1)
            + 1./4 * einsum('ijmc,mkab,kc...->ijab...', gooov, t2, r1)
            + 1./4 * einsum('keab,ijec,kc...->ijab...', govvv, t2, r1))
        return b21

    return _a12, _b12, _a21, _b21


def twobody_hessian(foo, fvv, goooo, govov, gvvvv, t2, disk=False):
    m1oo, m1vv = onebody_density(t2)
    mo, uo = scipy.linalg.eigh(m1oo)
    mv, uv = scipy.linalg.eigh(m1vv)
    uot = numpy.ascontiguousarray(numpy.transpose(uo))
    uvt = numpy.ascontiguousarray(numpy.transpose(uv))
    # oo
    tffoo = transform(foo, (uo, uo))
    tffoo /= (cast(mo, 0, 2) + cast(mo, 1, 2) - 1)
    ffoo = transform(tffoo, (uot, uot))
    # vv
    tffvv = transform(fvv, (uv, uv))
    tffvv /= (cast(mv, 0, 2) + cast(mv, 1, 2) - 1)
    ffvv = transform(tffvv, (uvt, uvt))
    # oooo
    no = len(mo)
    fgoooo = transform(goooo, (uo, uo, uo, uo))
    fgoooo[dix(no, (1, 2))] -= cast(tffoo, (0, 2))
    fgoooo[dix(no, (0, 3))] -= cast(tffoo, (1, 2))
    fgoooo /= (cast(mo, 0, 4) + cast(mo, 2, 4) - 1)
    fgoooo /= (cast(mo, 1, 4) + cast(mo, 3, 4) - 1)
    fgoooo = transform(fgoooo, (uot, uot, uot, uot))
    # ovov
    fgovov = transform(govov, (uo, uv, uo, uv))
    fgovov /= (cast(mo, 0, 4) + cast(mo, 2, 4) - 1)
    fgovov /= (cast(mv, 1, 4) + cast(mv, 3, 4) - 1)
    fgovov = transform(fgovov, (uot, uvt, uot, uvt))
    # vvvv
    nv = len(mv)
    fgvvvv = transform(gvvvv, (uv, uv, uv, uv))
    fgvvvv[dix(nv, (1, 2))] -= cast(tffvv, (0, 2))
    fgvvvv[dix(nv, (0, 3))] -= cast(tffvv, (1, 2))
    fgvvvv /= (cast(mv, 0, 4) + cast(mv, 2, 4) - 1)
    fgvvvv /= (cast(mv, 1, 4) + cast(mv, 3, 4) - 1)
    fgvvvv = transform(fgvvvv, (uvt, uvt, uvt, uvt))

    if disk:
        flname = tempfile.mkstemp()[1]
        fl = h5py.File(flname, mode='w')
        fgvvvv = fl.create_dataset('fgvvvv', data=fgvvvv)

    def _a22(r2):
        a22 = asm('0/1|2/3')(
            - 1./2 * einsum('ac,ijcb...->ijab...', ffvv, r2)
            - 1./2 * einsum('ik,kjab...->ijab...', ffoo, r2)
            + 1./8 * einsum('abcd,ijcd...->ijab...', gvvvv, r2)
            + 1./8 * einsum('ijkl,klab...->ijab...', goooo, r2)
            - einsum('jcla,ilcb...->ijab...', govov, r2)
            + 1./4 * einsum('afec,ijeb,klfd,klcd...->ijab...',
                            fgvvvv, t2, t2, r2)
            + 1./4 * einsum('kame,ijeb,mlcd,klcd...->ijab...',
                            fgovov, t2, t2, r2)
            + 1./4 * einsum('meic,mjab,kled,klcd...->ijab...',
                            fgovov, t2, t2, r2)
            + 1./4 * einsum('mkin,mjab,nlcd,klcd...->ijab...',
                            fgoooo, t2, t2, r2))
        return a22

    def _b22(r2):
        b22 = asm('0/1|2/3')(
            + 1./4 * einsum('acef,ijeb,klfd,klcd...->ijab...',
                            fgvvvv, t2, t2, r2)
            + 1./4 * einsum('nake,ijeb,nlcd,klcd...->ijab...',
                            fgovov, t2, t2, r2)
            + 1./4 * einsum('mcif,mjab,klfd,klcd...->ijab...',
                            fgovov, t2, t2, r2)
            + 1./4 * einsum('mnik,mjab,nlcd,klcd...->ijab...',
                            fgoooo, t2, t2, r2))
        return b22

    return _a22, _b22


def onebody_metric(t2):
    m1oo, m1vv = onebody_density(t2)

    def _s11(r1):
        return (
            + einsum('ij,ja...->ia...', m1oo, r1)
            - einsum('ab,ib...->ia...', m1vv, r1))

    return _s11


def onebody_metric_function(t2, f):
    m1oo, m1vv = onebody_density(t2)
    mo, uo = scipy.linalg.eigh(m1oo)
    mv, uv = scipy.linalg.eigh(m1vv)

    def _x11(r1):
        yovOV = einsum('iI,aA->iaIA', uo, uv)
        zovOV = einsum('iI,aA->IAia', uo, uv)
        yovOV *= f(cast(mo, 2, 4) - cast(mv, 3, 4))
        return einsum('iaJB,JBkc,kc...->ia...', yovOV, zovOV, r1)

    return _x11
