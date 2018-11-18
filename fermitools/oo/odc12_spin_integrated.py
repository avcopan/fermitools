import sys
import warnings
import numpy
import scipy.linalg

from .util import diis_extrapolator
from ..math import expm
from ..math import einsum
from ..math import cast
from ..math import transform
from ..math.asym import antisymmetrizer_product as asm


def solve(h_ao, r_ao, co_guess, cv_guess, t2_guess, maxiter=50, rthresh=1e-8,
          diis_start=3, diis_nvec=20, print_conv=True):
    Co_guess, CO_guess = co_guess
    Cv_guess, CV_guess = cv_guess
    No, NO, Nv, NV = (
            mo_dim for ao_dim, mo_dim in
            map(numpy.shape, (Co_guess, CO_guess, Cv_guess, CV_guess)))
    T1_a = numpy.zeros((No, Nv))
    T1_b = numpy.zeros((NO, NV))
    T2_aa = t2_guess[:No, :No, :Nv, :Nv]
    T2_ab = t2_guess[:No, No:, :Nv, Nv:]
    T2_bb = t2_guess[No:, No:, Nv:, Nv:]
    M1oo, M1OO, M1vv, M1VV = onebody_density_alt(T2_aa, T2_ab, T2_bb)

    trs = ()
    TRs = ()
    extrapolate = diis_extrapolator(start=diis_start, nvec=diis_nvec)

    for iteration in range(maxiter):
        Co, Cv = orbital_rotation_alt(Co_guess, Cv_guess, T1_a)
        CO, CV = orbital_rotation_alt(CO_guess, CV_guess, T1_b)
        Hoo = transform(h_ao, (Co, Co)) # <a|a>
        Hov = transform(h_ao, (Co, Cv))
        Hvv = transform(h_ao, (Cv, Cv))
        HOO = transform(h_ao, (CO, CO)) # <b|b>
        HOV = transform(h_ao, (CO, CV))
        HVV = transform(h_ao, (CV, CV))
        Goooo = asm("0/1")(transform(r_ao, (Co, Co, Co, Co))) # <aa|aa>
        Gooov = asm("0/1")(transform(r_ao, (Co, Co, Co, Cv)))
        Goovv = asm("0/1")(transform(r_ao, (Co, Co, Cv, Cv)))
        Govov = (transform(r_ao, (Co, Cv, Co, Cv))
                 - numpy.swapaxes(transform(r_ao, (Co, Cv, Cv, Co)), 2, 3))
        Govvv = asm("2/3")(transform(r_ao, (Co, Cv, Cv, Cv)))
        Gvvvv = asm("2/3")(transform(r_ao, (Cv, Cv, Cv, Cv)))
        GoOoO = transform(r_ao, (Co, CO, Co, CO))             # <ab|ab>
        GoOoV = transform(r_ao, (Co, CO, Co, CV))
        GoOvV = transform(r_ao, (Co, CO, Cv, CV))
        GoVoV = transform(r_ao, (Co, CV, Co, CV))
        GoVvV = transform(r_ao, (Co, CV, Cv, CV))
        GvVvV = transform(r_ao, (Cv, CV, Cv, CV))
        #
        GOoOv = transform(r_ao, (CO, Co, CO, Cv))
        GOvOv = transform(r_ao, (CO, Cv, CO, Cv))
        GOvVv = transform(r_ao, (CO, Cv, CV, Cv))
        GvOoV = transform(r_ao, (Cv, CO, Co, CV))
        # ^ extras
        GOOOO = asm("0/1")(transform(r_ao, (CO, CO, CO, CO))) # <bb|bb>
        GOOOV = asm("0/1")(transform(r_ao, (CO, CO, CO, CV)))
        GOOVV = asm("0/1")(transform(r_ao, (CO, CO, CV, CV)))
        GOVOV = (transform(r_ao, (CO, CV, CO, CV))
                 - numpy.swapaxes(transform(r_ao, (CO, CV, CV, CO)), 2, 3))
        GOVVV = asm("2/3")(transform(r_ao, (CO, CV, CV, CV)))
        GVVVV = asm("2/3")(transform(r_ao, (CV, CV, CV, CV)))

        # Orbital step
        M1oo, M1OO, M1vv, M1VV = onebody_density_alt(T2_aa, T2_ab, T2_bb)
        Foo = (Hoo
               + numpy.tensordot(Goooo, M1oo, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GoOoO, M1OO, axes=((1, 3), (0, 1)))
               + numpy.tensordot(Govov, M1vv, axes=((1, 3), (0, 1)))
               + numpy.tensordot(GoVoV, M1VV, axes=((1, 3), (0, 1))))
        FOO = (HOO
               + numpy.tensordot(GoOoO, M1oo, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GOOOO, M1OO, axes=((1, 3), (0, 1)))
               + numpy.tensordot(GOvOv, M1vv, axes=((1, 3), (0, 1)))
               + numpy.tensordot(GOVOV, M1VV, axes=((1, 3), (0, 1))))
        Fov = (Hov
               + numpy.tensordot(Gooov, M1oo, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GOoOv, M1OO, axes=((0, 2), (0, 1)))
               + numpy.tensordot(Govvv, M1vv, axes=((1, 3), (0, 1)))
               + numpy.tensordot(GoVvV, M1VV, axes=((1, 3), (0, 1))))
        FOV = (HOV
               + numpy.tensordot(GoOoV, M1oo, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GOOOV, M1OO, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GOvVv, M1vv, axes=((1, 3), (0, 1)))
               + numpy.tensordot(GOVVV, M1VV, axes=((1, 3), (0, 1))))
        Fvv = (Hvv
               + numpy.tensordot(Govov, M1oo, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GOvOv, M1OO, axes=((0, 2), (0, 1)))
               + numpy.tensordot(Gvvvv, M1vv, axes=((1, 3), (0, 1)))
               + numpy.tensordot(GvVvV, M1VV, axes=((1, 3), (0, 1))))
        FVV = (HVV
               + numpy.tensordot(GoVoV, M1oo, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GOVOV, M1OO, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GvVvV, M1vv, axes=((0, 2), (0, 1)))
               + numpy.tensordot(GVVVV, M1VV, axes=((1, 3), (0, 1))))
        Eo, EO, Ev, EV = map(numpy.diagonal, (Foo, FOO, Fvv, FVV))
        E1_a = cast(Eo, 0, 2) - cast(Ev, 1, 2)
        E1_b = cast(EO, 0, 2) - cast(EV, 1, 2)
        R1_a = (+ einsum('ma,im->ia', Fov, M1oo)
                - einsum('ie,ae->ia', Fov, M1vv)
                - 1./2 * (einsum('mnie,mnae->ia', Gooov, T2_aa) +
                          einsum('mNiE,mNaE->ia', GoOoV, T2_ab) * 2)
                + 1./2 * (einsum('maef,mief->ia', Govvv, T2_aa) +
                          einsum('MaEf,iMfE->ia', GOvVv, T2_ab) * 2)
                + 1./4 * (einsum('mlna,mlcd,nicd->ia', Gooov, T2_aa, T2_aa) +
                          einsum('MlNa,lMdC,iNdC->ia', GOoOv, T2_ab, T2_ab) * 4)
                - 1./4 * (einsum('ifed,klaf,kled->ia', Govvv, T2_aa, T2_aa) +
                          einsum('iFeD,kLaF,kLeD->ia', GoVvV, T2_ab, T2_ab) * 4)
                - (+ einsum('mfae,ikfc,mkec->ia', Govvv, T2_aa, T2_aa)
                   - einsum('MfEa,ikfc,kMcE->ia', GOvVv, T2_aa, T2_ab)
                   - einsum('MfEa,iKfC,MKEC->ia', GOvVv, T2_ab, T2_bb)
                   + einsum('mfae,iKfC,mKeC->ia', Govvv, T2_ab, T2_ab)
                   + einsum('mFaE,iKcF,mKcE->ia', GoVvV, T2_ab, T2_ab))
                + (+ einsum('mine,nkac,mkec->ia', Gooov, T2_aa, T2_aa)
                   - einsum('iMnE,nkac,kMcE->ia', GoOoV, T2_aa, T2_ab)
                   - einsum('iMnE,nKaC,MKEC->ia', GoOoV, T2_ab, T2_bb)
                   + einsum('mine,nKaC,mKeC->ia', Gooov, T2_ab, T2_ab)
                   + einsum('MiNe,kNaC,kMeC->ia', GOoOv, T2_ab, T2_ab)))
        R1_b = (+ einsum('MA,IM->IA', FOV, M1OO)
                - einsum('IE,AE->IA', FOV, M1VV)
                - 1./2 * (einsum('MNIE,MNAE->IA', GOOOV, T2_bb) +
                          einsum('MnIe,nMeA->IA', GOoOv, T2_ab) * 2)
                + 1./2 * (einsum('MAEF,MIEF->IA', GOVVV, T2_bb) +
                          einsum('mAeF,mIeF->IA', GoVvV, T2_ab) * 2)
                + 1./4 * (einsum('MLNA,MLCD,NICD->IA', GOOOV, T2_bb, T2_bb) +
                          einsum('mLnA,mLcD,nIcD->IA', GoOoV, T2_ab, T2_ab) * 4)
                - 1./4 * (einsum('IFED,KLAF,KLED->IA', GOVVV, T2_bb, T2_bb) +
                          einsum('IfEd,lKfA,lKdE->IA', GOvVv, T2_ab, T2_ab) * 4)
                - (+ einsum('MFAE,IKFC,MKEC->IA', GOVVV, T2_bb, T2_bb)
                   - einsum('mFeA,IKFC,mKeC->IA', GoVvV, T2_bb, T2_ab)
                   - einsum('mFeA,kIcF,mkec->IA', GoVvV, T2_ab, T2_aa)
                   + einsum('MFAE,kIcF,kMcE->IA', GOVVV, T2_ab, T2_ab)
                   + einsum('MfAe,kIfC,kMeC->IA', GOvVv, T2_ab, T2_ab))
                + (+ einsum('MINE,NKAC,MKEC->IA', GOOOV, T2_bb, T2_bb)
                   - einsum('ImNe,NKAC,mKeC->IA', GOoOv, T2_bb, T2_ab)
                   - einsum('ImNe,kNcA,mkec->IA', GOoOv, T2_ab, T2_aa)
                   + einsum('MINE,kNcA,kMcE->IA', GOOOV, T2_ab, T2_ab)
                   + einsum('mInE,nKcA,mKcE->IA', GoOoV, T2_ab, T2_ab)))
        T1_a = T1_a + R1_a / E1_a
        T1_b = T1_b + R1_b / E1_b

        # Amplitude step
        fFoo = fancy_property(Foo, M1oo)
        fFOO = fancy_property(FOO, M1OO)
        fFvv = fancy_property(Fvv, M1vv)
        fFVV = fancy_property(FVV, M1VV)
        fEo = numpy.diagonal(fFoo)
        fEO = numpy.diagonal(fFOO)
        fEv = numpy.diagonal(fFvv)
        fEV = numpy.diagonal(fFVV)
        fE2_aa = (+ cast(fEo, 0, 4) + cast(fEo, 1, 4)
                  + cast(fEv, 2, 4) + cast(fEv, 3, 4))
        fE2_ab = (+ cast(fEo, 0, 4) + cast(fEO, 1, 4)
                  + cast(fEv, 2, 4) + cast(fEV, 3, 4))
        fE2_bb = (+ cast(fEO, 0, 4) + cast(fEO, 1, 4)
                  + cast(fEV, 2, 4) + cast(fEV, 3, 4))
        R2_aa = (Goovv
                 + asm("2/3")(einsum('ac,ijcb->ijab', -fFvv, T2_aa))
                 - asm("0/1")(einsum('ki,kjab->ijab', +fFoo, T2_aa))
                 + 1./2 * einsum("abcd,ijcd->ijab", Gvvvv, T2_aa)
                 + 1./2 * einsum("klij,klab->ijab", Goooo, T2_aa)
                 - asm("0/1|2/3")(einsum("kaic,jkbc->ijab", Govov, T2_aa))
                 + asm("0/1|2/3")(einsum("aKiC,jKbC->ijab", GvOoV, T2_ab)))
        R2_ab = (GoOvV
                 + einsum('ac,iJcB->iJaB', -fFvv, T2_ab)
                 + einsum('BC,iJaC->iJaB', -fFVV, T2_ab)
                 - einsum('ki,kJaB->iJaB', +fFoo, T2_ab)
                 - einsum('KJ,iKaB->iJaB', +fFOO, T2_ab)
                 + einsum("aBcD,iJcD->iJaB", GvVvV, T2_ab)
                 + einsum("kLiJ,kLaB->iJaB", GoOoO, T2_ab)
                 - einsum("kaic,kJcB->iJaB", Govov, T2_ab)
                 + einsum("aKiC,JKBC->iJaB", GvOoV, T2_bb)
                 - einsum("KaJc,iKcB->iJaB", GOvOv, T2_ab)
                 - einsum("kBiC,kJaC->iJaB", GoVoV, T2_ab)
                 + einsum("cJkB,ikac->iJaB", GvOoV, T2_aa)
                 - einsum("KBJC,iKaC->iJaB", GOVOV, T2_ab))
        R2_bb = (GOOVV
                 + asm("2/3")(einsum('AC,IJCB->IJAB', -fFVV, T2_bb))
                 - asm("0/1")(einsum('KI,KJAB->IJAB', +fFOO, T2_bb))
                 + 1./2 * einsum("ABCD,IJCD->IJAB", GVVVV, T2_bb)
                 + 1./2 * einsum("KLIJ,KLAB->IJAB", GOOOO, T2_bb)
                 - asm("0/1|2/3")(einsum("KAIC,JKBC->IJAB", GOVOV, T2_bb))
                 + asm("0/1|2/3")(einsum("cIkA,kJcB->IJAB", GvOoV, T2_ab)))
        T2_aa = T2_aa + R2_aa / fE2_aa
        T2_ab = T2_ab + R2_ab / fE2_ab
        T2_bb = T2_bb + R2_bb / fE2_bb

        r1max = max(numpy.amax(numpy.abs(R1_a)),
                    numpy.amax(numpy.abs(R1_b)))
        r2max = max(numpy.amax(numpy.abs(R2_aa)),
                    numpy.amax(numpy.abs(R2_ab)),
                    numpy.amax(numpy.abs(R2_bb)))

        info = {'niter': iteration + 1, 'r1max': r1max, 'r2max': r2max}

        converged = r1max < rthresh and r2max < rthresh

        if print_conv:
            print(info)
            sys.stdout.flush()

        if converged:
            break

        # replace R2_ab <=> 2*R2_ab to match the spin-orbital case exactly
        (T1_a, T1_b, T2_aa, T2_ab, T2_bb), TRs = extrapolate(
            t=(T1_a, T1_b, T2_aa, T2_ab, T2_bb),
            r=(R1_a, R1_b, R2_aa, R2_ab, R2_bb),
            trs=TRs)

    # calculate the energy
    Foo = (Hoo
           + numpy.tensordot(Goooo, M1oo, axes=((0, 2), (0, 1)))
           + numpy.tensordot(GoOoO, M1OO, axes=((1, 3), (0, 1)))
           + numpy.tensordot(Govov, M1vv, axes=((1, 3), (0, 1)))
           + numpy.tensordot(GoVoV, M1VV, axes=((1, 3), (0, 1))))
    FOO = (HOO
           + numpy.tensordot(GoOoO, M1oo, axes=((0, 2), (0, 1)))
           + numpy.tensordot(GOOOO, M1OO, axes=((1, 3), (0, 1)))
           + numpy.tensordot(GOvOv, M1vv, axes=((1, 3), (0, 1)))
           + numpy.tensordot(GOVOV, M1VV, axes=((1, 3), (0, 1))))
    Fvv = (Hvv
           + numpy.tensordot(Govov, M1oo, axes=((0, 2), (0, 1)))
           + numpy.tensordot(GOvOv, M1OO, axes=((0, 2), (0, 1)))
           + numpy.tensordot(Gvvvv, M1vv, axes=((1, 3), (0, 1)))
           + numpy.tensordot(GvVvV, M1VV, axes=((1, 3), (0, 1))))
    FVV = (HVV
           + numpy.tensordot(GoVoV, M1oo, axes=((0, 2), (0, 1)))
           + numpy.tensordot(GOVOV, M1OO, axes=((0, 2), (0, 1)))
           + numpy.tensordot(GvVvV, M1vv, axes=((0, 2), (0, 1)))
           + numpy.tensordot(GVVVV, M1VV, axes=((1, 3), (0, 1))))
    en_elec = (
               + 1./2 * (+ numpy.vdot(Hoo + Foo, M1oo)
                         + numpy.vdot(HOO + FOO, M1OO))
               + 1./2 * (+ numpy.vdot(Hvv + Fvv, M1vv)
                         + numpy.vdot(HVV + FVV, M1VV))
               + 1./2 * (+ numpy.vdot(Goovv, T2_aa)
                         + numpy.vdot(GOOVV, T2_bb)
                         + numpy.vdot(GoOvV, T2_ab) * 4)
               + 1./8 * (+ einsum('abcd,klab,klcd', Gvvvv, T2_aa, T2_aa)
                         + einsum('ABCD,KLAB,KLCD', GVVVV, T2_bb, T2_bb)
                         + einsum('aBcD,kLaB,kLcD', GvVvV, T2_ab, T2_ab) * 8)
               + 1./8 * (+ einsum('ijkl,ijcd,klcd', Goooo, T2_aa, T2_aa)
                         + einsum('IJKL,IJCD,KLCD', GOOOO, T2_bb, T2_bb)
                         + einsum('iJkL,iJcD,kLcD', GoOoO, T2_ab, T2_ab) * 8)
               - (+ einsum('iajb,jkac,ikbc', Govov, T2_aa, T2_aa)
                  + einsum('iajb,jKaC,iKbC', Govov, T2_ab, T2_ab)
                  + einsum('IAJB,JKAC,IKBC', GOVOV, T2_bb, T2_bb)
                  + einsum('IAJB,kJcA,kIcB', GOVOV, T2_ab, T2_ab)
                  + einsum('iAjB,jKcA,iKcB', GoVoV, T2_ab, T2_ab)
                  + einsum('IaJb,kJaC,kIbC', GOvOv, T2_ab, T2_ab)
                  - einsum('aIjB,jkac,kIcB', GvOoV, T2_aa, T2_ab)
                  - einsum('aIjB,jKaC,IKBC', GvOoV, T2_ab, T2_bb)
                  - einsum('aIjB,jKaC,KICB', GvOoV, T2_ab, T2_bb)
                  - einsum('aIjB,jkac,kIcB', GvOoV, T2_aa, T2_ab))
           )

    co = Co, CO
    cv = Cv, CV
    t1, t2 = recover_spinorbital_amplitudes(T1_a, T1_b, T2_aa, T2_ab, T2_bb)

    if not converged:
        warnings.warn("Did not converge!")
    return en_elec, co, cv, t2, info


def fancy_property(pxx, m1xx):
    """ p_p^q (d m^p_q / d t) -> fp_p^q (d k^px_qx / dt)

    The one-body operator p can have multiple components.  Its spin-orbital
    indices are assumed to be the last two axes of the array.

    :param pxx: occupied or virtual block of a one-body operator
    :type pxx: numpy.ndarray
    :param m1xx: occupied or virtual block of the density
    :type m1xx: numpy.ndarray

    :returns: the derivative trace intermediate
    :rtype: numpy.ndarray
    """
    mx, ux = scipy.linalg.eigh(m1xx)
    ndim = pxx.ndim
    n1xx = cast(mx, ndim-2, ndim) + cast(mx, ndim-1, ndim) - 1
    tfpxx = transform(pxx, (ux, ux)) / n1xx
    uxt = numpy.ascontiguousarray(numpy.transpose(ux))
    fpxx = transform(tfpxx, (uxt, uxt))
    return fpxx


def onebody_density_alt(T2_aa, T2_ab, T2_bb):
    Doo = -1./2 * (einsum('ikcd,jkcd->ij', T2_aa, T2_aa) +
                   einsum('iKcD,jKcD->ij', T2_ab, T2_ab) * 2)
    DOO = -1./2 * (einsum('IKCD,JKCD->IJ', T2_bb, T2_bb) +
                   einsum('kIdC,kJdC->IJ', T2_ab, T2_ab) * 2)
    Dvv = -1./2 * (einsum('klac,klbc->ab', T2_aa, T2_aa) +
                   einsum('kLaC,kLbC->ab', T2_ab, T2_ab) * 2)
    DVV = -1./2 * (einsum('KLAC,KLBC->AB', T2_bb, T2_bb) +
                   einsum('lKcA,lKcB->AB', T2_ab, T2_ab) * 2)
    Ioo = numpy.eye(*Doo.shape)
    IOO = numpy.eye(*DOO.shape)
    Ivv = numpy.eye(*Dvv.shape)
    IVV = numpy.eye(*DVV.shape)
    M1oo = 1./2 * Ioo + numpy.real(scipy.linalg.sqrtm(Doo + 1./4 * Ioo))
    M1OO = 1./2 * IOO + numpy.real(scipy.linalg.sqrtm(DOO + 1./4 * IOO))
    M1vv = 1./2 * Ivv - numpy.real(scipy.linalg.sqrtm(Dvv + 1./4 * Ivv))
    M1VV = 1./2 * IVV - numpy.real(scipy.linalg.sqrtm(DVV + 1./4 * IVV))
    return M1oo, M1OO, M1vv, M1VV


def orbital_rotation_alt(Co, Cv, T1):
    No, Nv = numpy.shape(T1)
    Zoo = numpy.zeros((No, No))
    Zvv = numpy.zeros((Nv, Nv))
    A = numpy.bmat([[Zoo, -T1], [+T1.T, Zvv]])
    U = expm(A)
    C = numpy.hstack((Co, Cv))
    C = numpy.dot(C, U)
    Co, Cv = numpy.hsplit(C, (No,))
    return Co, Cv


def recover_spinorbital_amplitudes(T1_a, T1_b, T2_aa, T2_ab, T2_bb):
    No, NO, Nv, NV = numpy.shape(T2_ab)
    no = No + NO
    nv = Nv + NV
    t1 = numpy.zeros((no, nv))
    t2 = numpy.zeros((no, no, nv, nv))
    t1[:No, :Nv] = T1_a
    t1[No:, Nv:] = T1_b
    t2[:No, No:, :Nv, Nv:] = T2_ab
    t2 = asm("0/1|2/3")(t2)
    t2[:No, :No, :Nv, :Nv] = T2_aa
    t2[No:, No:, Nv:, Nv:] = T2_bb
    return t1, t2
