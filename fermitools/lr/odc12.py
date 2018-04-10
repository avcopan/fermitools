import numpy
import scipy

import sys
import time
# import h5py
# import tempfile

from toolz import functoolz
from .linmap import eye, add, subtract
from .blocker import build_block_vec
from .blocker import build_block_linmap
from .blocker import build_block_diag_linmap
from .diskdave import eig as eig_disk
from .coredave import eig as eig_core
from ..math import cast
from ..math import einsum
from ..math import transform
from ..math import diagonal_indices as dix
from ..math.asym import antisymmetrizer_product as asm
from ..math.spinorb import transform_onebody, transform_twobody

from ..math.direct import solve

from ..oo.odc12 import fock_xy
from ..oo.odc12 import fancy_property
from ..oo.odc12 import onebody_density


def solve_static_response(h_ao, p_ao, r_ao, co, cv, t2, maxdim=None,
                          maxiter=20, rthresh=1e-5, print_conv=False):
    hoo = transform_onebody(h_ao, (co, co))
    hov = transform_onebody(h_ao, (co, cv))
    hvv = transform_onebody(h_ao, (cv, cv))
    poo = transform_onebody(p_ao, (co, co))
    pov = transform_onebody(p_ao, (co, cv))
    pvv = transform_onebody(p_ao, (cv, cv))
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

    ioo, ivv = mixed_interaction(fov, gooov, govvv, m1oo, m1vv)

    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)
    fioo = fancy_property(ioo, m1oo)
    fivv = fancy_property(ivv, m1vv)
    fpoo = fancy_property(poo, m1oo)
    fpvv = fancy_property(pvv, m1vv)
    fgoooo, fgovov, fgvvvv = fancy_repulsion(
            foo, fvv, goooo, govov, gvvvv, m1oo, m1vv)

    pg1 = onebody_property_gradient(pov, m1oo, m1vv)
    pg2 = twobody_property_gradient(fpoo, -fpvv, t2)

    ad1 = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2 = twobody_hessian_zeroth_order_diagonal(foo, fvv, t2)

    a11, b11 = onebody_hessian(foo, fvv, goooo, goovv, govov, gvvvv, t2)
    a12, b12, a21, b21 = mixed_hessian(fioo, fivv, gooov, govvv, t2)
    a22, b22 = twobody_hessian(ffoo, ffvv, goooo, govov, gvvvv, fgoooo,
                               fgovov, fgvvvv, t2)

    no, _, nv, _ = t2.shape
    pg = build_block_vec(no, nv, pg1, pg2)
    ad = build_block_vec(no, nv, ad1, ad2)
    a = build_block_linmap(no, nv, l11=a11, l12=a12, l21=a21, l22=a22)
    b = build_block_linmap(no, nv, l11=b11, l12=b12, l21=b21, l22=b22)

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

    ioo, ivv = mixed_interaction(fov, gooov, govvv, m1oo, m1vv)

    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)
    fioo = fancy_property(ioo, m1oo)
    fivv = fancy_property(ivv, m1vv)
    fgoooo, fgovov, fgvvvv = fancy_repulsion(
            foo, fvv, goooo, govov, gvvvv, m1oo, m1vv)

    ad1 = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2 = twobody_hessian_zeroth_order_diagonal(foo, fvv, t2)

    a11, b11 = onebody_hessian(foo, fvv, goooo, goovv, govov, gvvvv, t2)
    a12, b12, a21, b21 = mixed_hessian(fioo, fivv, gooov, govvv, t2)
    a22, b22 = twobody_hessian(ffoo, ffvv, goooo, govov, gvvvv, fgoooo,
                               fgovov, fgvvvv, t2)

    s11 = onebody_metric(t2)
    sir11 = onebody_metric_function(
            t2, f=functoolz.compose(numpy.reciprocal, numpy.sqrt))

    no, _, nv, _ = t2.shape
    ad = build_block_vec(no, nv, ad1, ad2)
    a = build_block_linmap(no, nv, l11=a11, l12=a12, l21=a21, l22=a22)
    b = build_block_linmap(no, nv, l11=b11, l12=b12, l21=b21, l22=b22)
    s = build_block_diag_linmap(no, nv, l11=s11, l22=eye)
    sir = build_block_diag_linmap(no, nv, l11=sir11, l22=eye)

    h_plus = functoolz.compose(sir, add(a, b), sir)
    h_minus = functoolz.compose(sir, subtract(a, b), sir)

    h_bar = functoolz.compose(h_minus, h_plus)
    hd = ad * ad

    tm = time.time()
    if disk:
        w2, c_plus, info = eig_disk(
                a=h_bar, k=nroot, ad=hd, nconv=nconv, blsize=blsize,
                nguess=nguess, maxdim=maxdim, maxiter=maxiter, tol=rthresh,
                print_conv=print_conv, printf=numpy.sqrt)
    else:
        w2, c_plus, info = eig_core(
                a=h_bar, k=nroot, ad=hd, nconv=nconv, blsize=blsize,
                nguess=nguess, maxdim=maxdim, maxiter=maxiter, tol=rthresh,
                print_conv=print_conv, printf=numpy.sqrt)

    w = numpy.real(numpy.sqrt(w2))
    c_plus = numpy.real(c_plus)

    print("\nODC-12 excitation energies (in a.u.):")
    print(w.reshape(-1, 1))
    print("\nODC-12 excitation energies (in eV):")
    print(w.reshape(-1, 1)*27.2114)
    print('\nODC-12 linear response total time: {:8.1f}s'
          .format(time.time() - tm))
    sys.stdout.flush()

    x = sir(c_plus + h_plus(c_plus) / cast(w, 1, 2)) / 2.
    y = sir(c_plus - h_plus(c_plus) / cast(w, 1, 2)) / 2.

    if p_ao is not None:
        poo = transform_onebody(p_ao, (co, co))
        pov = transform_onebody(p_ao, (co, cv))
        pvv = transform_onebody(p_ao, (cv, cv))

        fpoo = fancy_property(poo, m1oo)
        fpvv = fancy_property(pvv, m1vv)

        pg1 = onebody_property_gradient(pov, m1oo, m1vv)
        pg2 = twobody_property_gradient(fpoo, -fpvv, t2)

        pg = build_block_vec(no, nv, pg1, pg2)

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


def mixed_hessian(fioo, fivv, gooov, govvv, t2):

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


def twobody_hessian(ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov,
                    fgvvvv, t2):

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


def fancy_repulsion(foo, fvv, goooo, govov, gvvvv, m1oo, m1vv):
    mo, uo = scipy.linalg.eigh(m1oo)
    mv, uv = scipy.linalg.eigh(m1vv)
    uot = numpy.ascontiguousarray(numpy.transpose(uo))
    uvt = numpy.ascontiguousarray(numpy.transpose(uv))
    no = len(mo)
    nv = len(mv)
    # tffoo
    tffoo = transform(foo, (uo, uo))
    tffoo /= (cast(mo, 0, 2) + cast(mo, 1, 2) - 1)
    # tffvv
    tffvv = transform(fvv, (uv, uv))
    tffvv /= (cast(mv, 0, 2) + cast(mv, 1, 2) - 1)
    # fgoooo
    fgoooo = transform(goooo, (uo, uo, uo, uo))
    fgoooo[dix(no, (1, 2))] -= cast(tffoo, (0, 2))
    fgoooo[dix(no, (0, 3))] -= cast(tffoo, (1, 2))
    fgoooo /= (cast(mo, 0, 4) + cast(mo, 2, 4) - 1)
    fgoooo /= (cast(mo, 1, 4) + cast(mo, 3, 4) - 1)
    fgoooo = transform(fgoooo, (uot, uot, uot, uot))
    # fgovov
    fgovov = transform(govov, (uo, uv, uo, uv))
    fgovov /= (cast(mo, 0, 4) + cast(mo, 2, 4) - 1)
    fgovov /= (cast(mv, 1, 4) + cast(mv, 3, 4) - 1)
    fgovov = transform(fgovov, (uot, uvt, uot, uvt))
    # fgvvvv
    fgvvvv = transform(gvvvv, (uv, uv, uv, uv))
    fgvvvv[dix(nv, (1, 2))] -= cast(tffvv, (0, 2))
    fgvvvv[dix(nv, (0, 3))] -= cast(tffvv, (1, 2))
    fgvvvv /= (cast(mv, 0, 4) + cast(mv, 2, 4) - 1)
    fgvvvv /= (cast(mv, 1, 4) + cast(mv, 3, 4) - 1)
    fgvvvv = transform(fgvvvv, (uvt, uvt, uvt, uvt))
    return fgoooo, fgovov, fgvvvv


def mixed_interaction(fov, gooov, govvv, m1oo, m1vv):
    no, nv = fov.shape
    ioo = numpy.ascontiguousarray(
           - einsum('mlka,im->iakl', gooov, m1oo)
           + einsum('ilke,ae->iakl', gooov, m1vv))
    ivv = numpy.ascontiguousarray(
           + einsum('mcad,im->iadc', govvv, m1oo)
           - einsum('iced,ae->iadc', govvv, m1vv))
    # ioo_iakl * delta_ik += fov_la
    # ivv_iadc * delta_ac -= fov_id
    ioo[dix(no, (0, 2))] += cast(fov, (2, 1))
    ivv[dix(nv, (1, 3))] -= cast(fov, (1, 2))
    return ioo, ivv
