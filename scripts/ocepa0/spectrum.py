import numpy
import time

import fermitools
import interfaces.psi4 as interface

CHARGE = +0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

# Ground state options
OO_NITER = 200      # number of iterations
OO_RTHRESH = 1e-10  # convergence threshold

# Excited state options
LR_NROOT = 7        # number of roots
LR_NGUESS = 2       # number of guess vectors per root
LR_NVEC = 20        # number of subspace vectors per root
LR_NITER = 200      # number of iterations
LR_RTHRESH = 1e-11  # convergence threshold


def main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nbf = interface.integrals.nbf(BASIS, LABELS)
    no = na + nb
    nv = 2*nbf - no

    # Integrals
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    # Mean-field guess orbitals
    c_guess = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    t2_guess = numpy.zeros((no, no, nv, nv))

    # Solve ground state
    t = time.time()
    en_elec, c, t2, info = fermitools.oo.ocepa0.solve(
            na=na, nb=nb, h_ao=h_ao, r_ao=r_ao, c_guess=c_guess,
            t2_guess=t2_guess, niter=OO_NITER, r_thresh=OO_RTHRESH)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))
    print('time: {:8.1f}s'.format(time.time() - t))

    # LR inputs
    ac, bc = c
    aco, acv = numpy.split(ac, (na,), axis=1)
    bco, bcv = numpy.split(bc, (nb,), axis=1)
    co = (aco, bco)
    cv = (acv, bcv)
    hoo = fermitools.math.spinorb.transform_onebody(h_ao, (co, co))
    hov = fermitools.math.spinorb.transform_onebody(h_ao, (co, cv))
    hvv = fermitools.math.spinorb.transform_onebody(h_ao, (cv, cv))
    goooo = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, co))
    gooov = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, cv))
    goovv = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, cv, cv))
    govov = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, co, cv))
    govvv = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, cv, cv))
    gvvvv = fermitools.math.spinorb.transform_twobody(r_ao, (cv, cv, cv, cv))
    foo = fermitools.oo.ocepa0.fock_xy(hxy=hoo, goxoy=goooo)
    fov = fermitools.oo.ocepa0.fock_xy(hxy=hov, goxoy=gooov)
    fvv = fermitools.oo.ocepa0.fock_xy(hxy=hvv, goxoy=govov)

    sd = fermitools.lr.ocepa0.metric_zeroth_order_diagonal(no, nv)
    ad = fermitools.lr.ocepa0.hessian_zeroth_order_diagonal(foo=foo, fvv=fvv)

    s, d = fermitools.lr.ocepa0.metric(t2=t2)
    a, b = fermitools.lr.ocepa0.hessian(
            foo=foo, fov=fov, fvv=fvv, goooo=goooo, gooov=gooov, goovv=goovv,
            govov=govov, govvv=govvv, gvvvv=gvvvv, t2=t2)

    t = time.time()
    w, u, info = fermitools.lr.solve.spectrum(
            a=a, b=b, s=s, d=d, ad=ad, sd=sd, nroot=LR_NROOT, nguess=LR_NGUESS,
            nvec=LR_NVEC, niter=LR_NITER, r_thresh=LR_RTHRESH)
    print(w)
    print('time: {:8.1f}s'.format(time.time() - t))


if __name__ == '__main__':
    main()
