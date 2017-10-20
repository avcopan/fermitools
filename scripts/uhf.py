import numpy
import scipy.linalg as spla

import fermitools

import interfaces.psi4 as interface


def solve_uhf(oa, ob, s, h, r, d_guess=None, niter=50, e_thresh=1e-10):
    if d_guess is None:
        ad = bd = numpy.zeros_like(s)
    else:
        ad, bd = d_guess

    en_elec_last = 0.
    for iter_ in range(niter):
        af, bf = fermitools.scf.uhf.fock(h, r, ad, bd)
        ae, ac = spla.eigh(af, s)
        be, bc = spla.eigh(bf, s)
        ad = fermitools.scf.density(ac[:, oa])
        bd = fermitools.scf.density(bc[:, ob])

        af, bf = fermitools.scf.uhf.fock(h, r, ad, bd)

        en_elec = fermitools.scf.uhf.energy(h, af, bf, ad, bd)
        en_change = en_elec - en_elec_last
        en_elec_last = en_elec

        # print('@UHF {:<3d} {:20.15f} {:20.15f}'.format(iter_, en_elec,
        #                                                en_change))

        converged = (numpy.fabs(en_change) < e_thresh)

        if converged:
            break

    return en_elec, numpy.array([ac, bc])


def energy_routine(basis, labels, coords, charge, spin,
                   niter=50, e_thresh=1e-12, return_coeffs=False):
    # Spaces
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    oa = slice(None, na)
    ob = slice(None, nb)

    # Integrals
    s = interface.integrals.overlap(basis, labels, coords)
    h = interface.integrals.core_hamiltonian(basis, labels, coords)
    r = interface.integrals.repulsion(basis, labels, coords)

    # Call the solver
    en_elec, c = solve_uhf(oa, ob, s, h, r, e_thresh=1e-14)

    return en_elec if not return_coeffs else (en_elec, c)


def energy_function(basis, labels, coords, charge, spin,
                    niter=50, e_thresh=1e-12):
    # Spaces
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    oa = slice(None, na)
    ob = slice(None, nb)

    # Integrals
    s = interface.integrals.overlap(basis, labels, coords)
    m = interface.integrals.dipole(basis, labels, coords)
    h = interface.integrals.core_hamiltonian(basis, labels, coords)
    r = interface.integrals.repulsion(basis, labels, coords)

    # Call the solver
    def electronic_energy(f=(0., 0., 0.)):
        h_pert = h - numpy.tensordot(f, m, axes=(0, 0))
        en_elec, (ac, bc) = solve_uhf(oa, ob, s, h_pert, r, e_thresh=1e-14)
        return en_elec

    return electronic_energy


def main():
    CHARGE = +1
    SPIN = 1
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_elec, (ac, bc) = energy_routine(BASIS, LABELS, COORDS, CHARGE, SPIN,
                                       return_coeffs=True, e_thresh=1e-15)
    en_tot = en_elec + en_nuc

    print('{:20.15f}'.format(en_tot))

    # Evaluate dipole moment as expectation value
    m = interface.integrals.dipole(BASIS, LABELS, COORDS)
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    ad = fermitools.scf.density(ac[:, :na])
    bd = fermitools.scf.density(bc[:, :nb])
    mu = numpy.array([numpy.vdot(mx, ad) + numpy.vdot(mx, bd) for mx in m])

    # Evaluate dipole moment as energy derivative
    en_fn = energy_function(BASIS, LABELS, COORDS, CHARGE, SPIN,
                            e_thresh=1e-15)
    gr = fermitools.math.central_difference(en_fn, (0., 0., 0.),
                                            step=0.0025, npts=13)

    print(mu.round(11))
    print(gr.round(11))

    # Tests
    from numpy.testing import assert_almost_equal
    assert_almost_equal(en_tot, -74.66178436045595, decimal=10)
    assert_almost_equal(gr, -mu, decimal=8)

    def en_fn_alt(f):
        return en_fn((0., 0., f))

    gr_alt = fermitools.math.central_difference(en_fn_alt, 0., npts=13)
    print(gr_alt)


if __name__ == '__main__':
    main()
