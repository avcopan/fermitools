import numpy
import scipy

import sys
import time
import tempfile

from toolz import functoolz
from .linmap import eye, add, subtract
from .blocker import count_excitations
from .blocker import build_block_vec
from .blocker import build_block_linmap
from .blocker import build_block_diag_linmap
from .diskdave import dataset, remove_dataset, file_name
from .diskdave import eig as eig_disk
from .coredave import eig as eig_core
from ..math import cast
from ..math import einsum
from ..math import transform
from ..math import diagonal_indices as dix
from ..math.asym import symmetrizer_product as sm
from ..math.asym import antisymmetrizer_product as asm
from ..math.spinorb import transform_onebody, transform_twobody

from ..math.direct import solve

from ..oo.odc12 import fock_xy
from ..oo.odc12 import fancy_property
from ..oo.odc12 import onebody_density
from ..oo.odc12 import orbital_gradient_intermediate_xo
from ..oo.odc12 import orbital_gradient_intermediate_xv


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

    cfoo = orbital_gradient_intermediate_xo(fox=foo, gooox=goooo, goxvv=goovv,
                                            govxv=govov, t2=t2, m1oo=m1oo)
    cfvv = orbital_gradient_intermediate_xv(fxv=fvv, gooxv=goovv, goxov=govov,
                                            gxvvv=gvvvv, t2=t2, m1vv=m1vv)

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
    pg2 = twobody_property_gradient(fpoo, fpvv, t2)

    ad1z = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2z = twobody_hessian_zeroth_order_diagonal(ffoo, ffvv)

    a11, b11 = onebody_hessian(foo, fvv, cfoo, cfvv, goooo, goovv, govov,
                               gvvvv, t2, m1oo, m1vv)
    a12, b12, a21, b21 = mixed_hessian(fioo, fivv, gooov, govvv, t2)
    a22, b22 = twobody_hessian(ffoo, ffvv, goooo, govov, gvvvv, fgoooo,
                               fgovov, fgvvvv, t2)

    no, _, nv, _ = t2.shape
    pg = build_block_vec(no, nv, pg1, pg2)
    adz = build_block_vec(no, nv, ad1z, ad2z)
    a = build_block_linmap(no, nv, l11=a11, l12=a12, l21=a21, l22=a22)
    b = build_block_linmap(no, nv, l11=b11, l12=b12, l21=b21, l22=b22)

    print("Second-order (static) properties:")
    e = add(a, b)
    v = -2*pg
    r, info = solve(a=e, b=v, ad=adz, maxdim=maxdim, tol=rthresh,
                    print_conv=True)
    alpha = numpy.dot(r.T, pg)
    print(alpha.round(12))
    return alpha


def solve_spectrum(h_ao, r_ao, co, cv, t2, nroot=1, nconv=None, nguess=None,
                   maxdim=None, maxiter=100, rthresh=1e-5, print_conv=False,
                   disk=False, blsize=None, p_ao=None, exact_diagonal=False):
    prefix = tempfile.mkstemp()[1] if disk else None

    hoo = transform_onebody(h_ao, (co, co))
    hov = transform_onebody(h_ao, (co, cv))
    hvv = transform_onebody(h_ao, (cv, cv))
    goooo = transform_twobody(r_ao, (co, co, co, co))
    gooov = transform_twobody(r_ao, (co, co, co, cv))
    goovv = transform_twobody(r_ao, (co, co, cv, cv))
    govov = transform_twobody(r_ao, (co, cv, co, cv))
    govvv = transform_twobody(r_ao, (co, cv, cv, cv))
    gvvvv = transform_twobody(r_ao, (cv, cv, cv, cv))
    gvvvv = (gvvvv if not disk else
             dataset(file_name(prefix, 'gvvvv'), data=gvvvv))

    m1oo, m1vv = onebody_density(t2)

    foo = fock_xy(hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fock_xy(hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fock_xy(hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)

    cfoo = orbital_gradient_intermediate_xo(fox=foo, gooox=goooo, goxvv=goovv,
                                            govxv=govov, t2=t2, m1oo=m1oo)
    cfvv = orbital_gradient_intermediate_xv(fxv=fvv, gooxv=goovv, goxov=govov,
                                            gxvvv=gvvvv, t2=t2, m1vv=m1vv)

    ioo, ivv = mixed_interaction(fov, gooov, govvv, m1oo, m1vv)

    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)

    fioo = fancy_property(ioo, m1oo)
    fivv = fancy_property(ivv, m1vv)

    fgoooo, fgovov, fgvvvv = fancy_repulsion(
            foo, fvv, goooo, govov, gvvvv, m1oo, m1vv)
    fgvvvv = (fgvvvv if not disk else
              dataset(file_name(prefix, 'fgvvvv'), data=fgvvvv))

    ad1z = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2z = twobody_hessian_zeroth_order_diagonal(ffoo, ffvv)

    ad1, bd1 = onebody_hessian_diagonal(foo, fvv, cfoo, cfvv, goooo, goovv,
                                        govov, gvvvv, t2, m1oo, m1vv)
    ad2, bd2 = twobody_hessian_diagonal(ffoo, ffvv, goooo, govov, gvvvv,
                                        fgoooo, fgovov, fgvvvv, t2)

    a11, b11 = onebody_hessian(foo, fvv, cfoo, cfvv, goooo, goovv, govov,
                               gvvvv, t2, m1oo, m1vv)
    a12, b12, a21, b21 = mixed_hessian(fioo, fivv, gooov, govvv, t2)
    a22, b22 = twobody_hessian(ffoo, ffvv, goooo, govov, gvvvv, fgoooo,
                               fgovov, fgvvvv, t2)

    sd1 = onebody_metric_diagonal(m1oo, m1vv)
    sd2 = numpy.ones_like(ad2)

    s11 = onebody_metric(m1oo, m1vv)
    sir11 = onebody_metric_function(
            m1oo, m1vv, f=functoolz.compose(numpy.reciprocal, numpy.sqrt))

    no, _, nv, _ = t2.shape
    adz = build_block_vec(no, nv, ad1z, ad2z)
    ad = build_block_vec(no, nv, ad1, ad2)
    bd = build_block_vec(no, nv, bd1, bd2)
    sd = build_block_vec(no, nv, sd1, sd2)
    a = build_block_linmap(no, nv, l11=a11, l12=a12, l21=a21, l22=a22)
    b = build_block_linmap(no, nv, l11=b11, l12=b12, l21=b21, l22=b22)
    s = build_block_diag_linmap(no, nv, l11=s11, l22=eye)
    sir = build_block_diag_linmap(no, nv, l11=sir11, l22=eye)

    h_plus = functoolz.compose(sir, add(a, b), sir)
    h_minus = functoolz.compose(sir, subtract(a, b), sir)

    h_bar = functoolz.compose(h_minus, h_plus)
    n1, n2 = count_excitations(no, nv)
    hd = adz * adz if not exact_diagonal else (ad - bd) * (ad + bd) / sd**2

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
        pg2 = twobody_property_gradient(fpoo, fpvv, t2)

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

    if disk:
        remove_dataset(gvvvv)
        remove_dataset(fgvvvv)

    return w, (x, y), info


def solve_spectrum1(h_ao, r_ao, co, cv, t2, nroot=1, nconv=None, nguess=None,
                    maxdim=None, maxiter=100, rthresh=1e-5, print_conv=False,
                    blsize=None, exact_diagonal=False):
    from .linmap import negative, bmat, block_diag

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

    cfoo = orbital_gradient_intermediate_xo(fox=foo, gooox=goooo, goxvv=goovv,
                                            govxv=govov, t2=t2, m1oo=m1oo)
    cfvv = orbital_gradient_intermediate_xv(fxv=fvv, gooxv=goovv, goxov=govov,
                                            gxvvv=gvvvv, t2=t2, m1vv=m1vv)

    ioo, ivv = mixed_interaction(fov, gooov, govvv, m1oo, m1vv)

    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)

    fioo = fancy_property(ioo, m1oo)
    fivv = fancy_property(ivv, m1vv)

    fgoooo, fgovov, fgvvvv = fancy_repulsion(
            foo, fvv, goooo, govov, gvvvv, m1oo, m1vv)

    ad1z = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2z = twobody_hessian_zeroth_order_diagonal(ffoo, ffvv)

    ad1, bd1 = onebody_hessian_diagonal(foo, fvv, cfoo, cfvv, goooo, goovv,
                                        govov, gvvvv, t2, m1oo, m1vv)
    ad2, bd2 = twobody_hessian_diagonal(ffoo, ffvv, goooo, govov, gvvvv,
                                        fgoooo, fgovov, fgvvvv, t2)

    a11, b11 = onebody_hessian(foo, fvv, cfoo, cfvv, goooo, goovv, govov,
                               gvvvv, t2, m1oo, m1vv)
    a12, b12, a21, b21 = mixed_hessian(fioo, fivv, gooov, govvv, t2)
    a22, b22 = twobody_hessian(ffoo, ffvv, goooo, govov, gvvvv, fgoooo,
                               fgovov, fgvvvv, t2)

    sd1 = onebody_metric_diagonal(m1oo, m1vv)
    sd2 = numpy.ones_like(ad2)

    s11 = onebody_metric(m1oo, m1vv)

    no, _, nv, _ = t2.shape
    adz = build_block_vec(no, nv, ad1z, ad2z)
    ad = build_block_vec(no, nv, ad1, ad2)
    sd = build_block_vec(no, nv, sd1, sd2)
    a = build_block_linmap(no, nv, l11=a11, l12=a12, l21=a21, l22=a22)
    b = build_block_linmap(no, nv, l11=b11, l12=b12, l21=b21, l22=b22)
    s = build_block_diag_linmap(no, nv, l11=s11, l22=eye)

    e = bmat([[a, b], [b, a]], 2)
    m = block_diag((s, negative(s)), (len(ad),))
    ed = (numpy.concatenate((+ad, +ad)) if exact_diagonal else
          numpy.concatenate((+adz, +adz)))
    md = (numpy.concatenate((+sd, -sd)) if exact_diagonal else
          numpy.ones_like(ed))

    tm = time.time()
    wr, z, info = eig_core(
            a=m, k=-nroot, ad=md, b=e, bd=ed, nconv=nconv, blsize=blsize,
            nguess=nguess, maxdim=maxdim, maxiter=maxiter, tol=rthresh,
            print_conv=print_conv, printf=numpy.sqrt, sym=True)
    w = numpy.reciprocal(wr)
    x, y = numpy.split(z, 2)

    print("\nODC-12 excitation energies (in a.u.):")
    print(w.reshape(-1, 1))
    print("\nODC-12 excitation energies (in eV):")
    print(w.reshape(-1, 1)*27.2114)
    print('\nODC-12 linear response total time: {:8.1f}s'
          .format(time.time() - tm))
    sys.stdout.flush()

    return w, (x, y), info


def solve_spectrum2(h_ao, r_ao, co, cv, t2, nroot=1, nconv=None, nguess=None,
                    maxdim=None, maxiter=100, rthresh=1e-5, print_conv=False,
                    blsize=None, exact_diagonal=False):
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

    cfoo = orbital_gradient_intermediate_xo(fox=foo, gooox=goooo, goxvv=goovv,
                                            govxv=govov, t2=t2, m1oo=m1oo)
    cfvv = orbital_gradient_intermediate_xv(fxv=fvv, gooxv=goovv, goxov=govov,
                                            gxvvv=gvvvv, t2=t2, m1vv=m1vv)

    ioo, ivv = mixed_interaction(fov, gooov, govvv, m1oo, m1vv)

    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)

    fioo = fancy_property(ioo, m1oo)
    fivv = fancy_property(ivv, m1vv)

    fgoooo, fgovov, fgvvvv = fancy_repulsion(
            foo, fvv, goooo, govov, gvvvv, m1oo, m1vv)

    ad1z = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2z = twobody_hessian_zeroth_order_diagonal(ffoo, ffvv)

    ad1, bd1 = onebody_hessian_diagonal(foo, fvv, cfoo, cfvv, goooo, goovv,
                                        govov, gvvvv, t2, m1oo, m1vv)
    ad2, bd2 = twobody_hessian_diagonal(ffoo, ffvv, goooo, govov, gvvvv,
                                        fgoooo, fgovov, fgvvvv, t2)

    a11, b11 = onebody_hessian(foo, fvv, cfoo, cfvv, goooo, goovv, govov,
                               gvvvv, t2, m1oo, m1vv)
    a12, b12, a21, b21 = mixed_hessian(fioo, fivv, gooov, govvv, t2)
    a22, b22 = twobody_hessian(ffoo, ffvv, goooo, govov, gvvvv, fgoooo,
                               fgovov, fgvvvv, t2)

    sd1 = onebody_metric_diagonal(m1oo, m1vv)
    sd2 = numpy.ones_like(ad2)

    si11 = onebody_metric_function(m1oo, m1vv, f=numpy.reciprocal)

    no, _, nv, _ = t2.shape
    adz = build_block_vec(no, nv, ad1z, ad2z)
    ad = build_block_vec(no, nv, ad1, ad2)
    bd = build_block_vec(no, nv, bd1, bd2)
    sd = build_block_vec(no, nv, sd1, sd2)
    a = build_block_linmap(no, nv, l11=a11, l12=a12, l21=a21, l22=a22)
    b = build_block_linmap(no, nv, l11=b11, l12=b12, l21=b21, l22=b22)
    si = build_block_diag_linmap(no, nv, l11=si11, l22=eye)

    h_plus = functoolz.compose(si, add(a, b))
    h_minus = functoolz.compose(si, subtract(a, b))

    h_bar = functoolz.compose(h_minus, h_plus)
    n1, n2 = count_excitations(no, nv)
    hd = adz * adz if not exact_diagonal else (ad - bd) * (ad + bd) / sd**2

    tm = time.time()
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

    x = (c_plus + h_plus(c_plus) / cast(w, 1, 2)) / 2.
    y = (c_plus - h_plus(c_plus) / cast(w, 1, 2)) / 2.

    return w, (x, y), info


def solve_spectrum3(h_ao, r_ao, co, cv, t2, nroot=1, nconv=None, nguess=None,
                    maxdim=None, maxiter=100, rthresh=1e-5, print_conv=False,
                    blsize=None, exact_diagonal=False):
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

    cfoo = orbital_gradient_intermediate_xo(fox=foo, gooox=goooo, goxvv=goovv,
                                            govxv=govov, t2=t2, m1oo=m1oo)
    cfvv = orbital_gradient_intermediate_xv(fxv=fvv, gooxv=goovv, goxov=govov,
                                            gxvvv=gvvvv, t2=t2, m1vv=m1vv)

    ioo, ivv = mixed_interaction(fov, gooov, govvv, m1oo, m1vv)

    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)

    fioo = fancy_property(ioo, m1oo)
    fivv = fancy_property(ivv, m1vv)

    fgoooo, fgovov, fgvvvv = fancy_repulsion(
            foo, fvv, goooo, govov, gvvvv, m1oo, m1vv)

    ad1z = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2z = twobody_hessian_zeroth_order_diagonal(ffoo, ffvv)

    ad1, bd1 = onebody_hessian_diagonal(foo, fvv, cfoo, cfvv, goooo, goovv,
                                        govov, gvvvv, t2, m1oo, m1vv)
    ad2, bd2 = twobody_hessian_diagonal(ffoo, ffvv, goooo, govov, gvvvv,
                                        fgoooo, fgovov, fgvvvv, t2)

    a11, b11 = onebody_hessian(foo, fvv, cfoo, cfvv, goooo, goovv, govov,
                               gvvvv, t2, m1oo, m1vv)
    a12, b12, a21, b21 = mixed_hessian(fioo, fivv, gooov, govvv, t2)
    a22, b22 = twobody_hessian(ffoo, ffvv, goooo, govov, gvvvv, fgoooo,
                               fgovov, fgvvvv, t2)

    sd1 = onebody_metric_diagonal(m1oo, m1vv)
    sd2 = numpy.ones_like(ad2)

    sir11 = onebody_metric_function(
            m1oo, m1vv, f=functoolz.compose(numpy.reciprocal, numpy.sqrt))

    no, _, nv, _ = t2.shape
    adz = build_block_vec(no, nv, ad1z, ad2z)
    ad = build_block_vec(no, nv, ad1, ad2)
    bd = build_block_vec(no, nv, bd1, bd2)
    sd = build_block_vec(no, nv, sd1, sd2)
    a = build_block_linmap(no, nv, l11=a11, l12=a12, l21=a21, l22=a22)
    b = build_block_linmap(no, nv, l11=b11, l12=b12, l21=b21, l22=b22)
    sir = build_block_diag_linmap(no, nv, l11=sir11, l22=eye)

    h_plus = functoolz.compose(sir, add(a, b), sir)
    h_minus = functoolz.compose(sir, subtract(a, b), sir)

    h_bar = functoolz.compose(h_minus, h_plus)
    n1, n2 = count_excitations(no, nv)
    hd = adz * adz if not exact_diagonal else (ad - bd) * (ad + bd) / sd**2

    tm = time.time()
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

    return w, (x, y), info


# The LR-ODC-12 equations:
def onebody_hessian_zeroth_order_diagonal(foo, fvv):
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    return - cast(eo, 0, 2) + cast(ev, 1, 2)


def twobody_hessian_zeroth_order_diagonal(ffoo, ffvv):
    efo = numpy.diagonal(ffoo)
    efv = numpy.diagonal(ffvv)
    return (- cast(efo, 0, 4) - cast(efo, 1, 4)
            - cast(efv, 2, 4) - cast(efv, 3, 4))


def onebody_property_gradient(pov, m1oo, m1vv):
    return (
        + einsum('im,...ma->ia...', m1oo, pov)
        - einsum('...ie,ea->ia...', pov, m1vv))


def twobody_property_gradient(fpoo, fpvv, t2):
    return (
        - asm('2/3')(
              einsum('...ac,ijcb->ijab...', fpvv, t2))
        - asm('0/1')(
              einsum('...ik,kjab->ijab...', fpoo, t2)))


def onebody_hessian_diagonal(foo, fvv, cfoo, cfvv, goooo, goovv, govov, gvvvv,
                             t2, m1oo, m1vv):
    eo = numpy.diag(foo)
    ev = numpy.diag(fvv)
    ceo = numpy.diag(cfoo)
    cev = numpy.diag(cfvv)
    m1o = numpy.diag(m1oo)
    m1v = numpy.diag(m1vv)

    a11d = (cast(eo, 0, 2) * cast(m1v, 1, 2)
            + cast(ev, 1, 2) * cast(m1o, 0, 2)
            - cast(cev, 1, 2) - cast(ceo, 0, 2))

    a11d += (
        - numpy.einsum('mana,mi,in->ia', govov, m1oo, m1oo)
        - numpy.einsum('ieif,af,ea->ia', govov, m1vv, m1vv)
        + 2. * numpy.einsum('iame,im,ae->ia', govov, m1oo, m1vv)
        - numpy.einsum('mini,nkac,mkac->ia', goooo, t2, t2)
        + 1./2 * numpy.einsum('mana,micd,nicd->ia', govov, t2, t2)
        + 1./2 * numpy.einsum('ieif,klae,klaf->ia', govov, t2, t2)
        - numpy.einsum('aeaf,ikec,ikfc->ia', gvvvv, t2, t2)
        - 2. * numpy.einsum('iame,mkac,ikec->ia', govov, t2, t2))

    b11d = (
        + 2. * numpy.einsum('iema,imae->ia', govov, t2)
        + 1./2 * numpy.einsum('iimn,mnaa->ia', goooo, t2)
        + 1./2 * numpy.einsum('efaa,iief->ia', gvvvv, t2)
        + 2. * numpy.einsum('imae,im,ea->ia', goovv, m1oo, m1vv)
        + numpy.einsum('mnaa,im,in->ia', goovv, m1oo, m1oo)
        + numpy.einsum('iief,ea,fa->ia', goovv, m1vv, m1vv)
        + 1./4 * numpy.einsum('mnaa,iicd,mncd->ia', goovv, t2, t2)
        - 2. * numpy.einsum('imae,mkec,ikac->ia', goovv, t2, t2)
        + 1./4 * numpy.einsum('iief,klef,klaa->ia', goovv, t2, t2))

    return a11d, b11d


def twobody_hessian_diagonal(ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov,
                             fgvvvv, t2):
    feo = numpy.diag(ffoo)
    fev = numpy.diag(ffvv)

    goo = numpy.diagonal(
            numpy.diagonal(goooo, axis1=0, axis2=2), axis1=0, axis2=1)
    gvv = numpy.diagonal(
            numpy.diagonal(gvvvv, axis1=0, axis2=2), axis1=0, axis2=1)
    gov = numpy.diagonal(
            numpy.diagonal(govov, axis1=0, axis2=2), axis1=0, axis2=1)

    a22d = (- cast(feo, 0, 4) - cast(feo, 1, 4)
            - cast(fev, 2, 4) - cast(fev, 3, 4)
            + cast(goo, (0, 1), 4) + cast(gvv, (2, 3), 4))

    a22d += sm('0/1|2/3')(
        - cast(gov, (0, 2), 4)
        + 1./2 * numpy.einsum('afea,ijeb,ijfb->ijab', fgvvvv, t2, t2)
        - 1./2 * numpy.einsum('afeb,ijeb,ijfa->ijab', fgvvvv, t2, t2)
        + 2 * numpy.einsum('iame,ijeb,mjab->ijab', fgovov, t2, t2)
        + 1./2 * numpy.einsum('miin,mjab,njab->ijab', fgoooo, t2, t2)
        - 1./2 * numpy.einsum('mijn,miab,njab->ijab', fgoooo, t2, t2))

    b22d = sm('0/1|2/3')(
        + 1/2. * numpy.einsum('aaef,ijeb,ijfb->ijab', fgvvvv, t2, t2)
        - 1/2. * numpy.einsum('baef,ijea,ijfb->ijab', fgvvvv, t2, t2)
        + 2 * numpy.einsum('maie,ijeb,mjab->ijab', fgovov, t2, t2)
        + 1/2. * numpy.einsum('mnii,mjab,njab->ijab', fgoooo, t2, t2)
        - 1/2. * numpy.einsum('mnij,mjab,niab->ijab', fgoooo, t2, t2))

    return a22d, b22d


def onebody_metric_diagonal(m1oo, m1vv):
    m1o = numpy.diag(m1oo)
    m1v = numpy.diag(m1vv)
    s11d = cast(m1o, 0, 2) - cast(m1v, 1, 2)
    return s11d


def onebody_hessian(foo, fvv, fcoo, fcvv, goooo, goovv, govov, gvvvv, t2,
                    m1oo, m1vv):
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


def onebody_metric(m1oo, m1vv):

    def _s11(r1):
        return (
            + einsum('ij,ja...->ia...', m1oo, r1)
            - einsum('ab,ib...->ia...', m1vv, r1))

    return _s11


def onebody_metric_function(m1oo, m1vv, f):
    mo, uo = scipy.linalg.eigh(m1oo)
    mv, uv = scipy.linalg.eigh(m1vv)

    def _x11(r1):
        yovOV = einsum('iI,aA->iaIA', uo, uv)
        zovOV = einsum('iI,aA->IAia', uo, uv)
        yovOV *= f(cast(mo, 2, 4) - cast(mv, 3, 4))
        return einsum('iaJB,JBkc,kc...->ia...', yovOV, zovOV, r1)

    return _x11


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
