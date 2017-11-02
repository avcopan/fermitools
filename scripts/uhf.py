import numpy
import scipy.linalg as spla

import fermitools

import interfaces.psi4 as interface


def solve(naocc, nbocc, s, h, r, d_guess=None, niter=50, e_thresh=1e-10):
    if d_guess is None:
        ad = bd = numpy.zeros_like(s)
    else:
        ad, bd = d_guess

    en_elec_last = 0.
    for iter_ in range(niter):
        af, bf = fermitools.scf.uhf.fock(h, r, ad, bd)
        ae, ac = spla.eigh(af, s)
        be, bc = spla.eigh(bf, s)
        ad = fermitools.scf.density(ac[:, :naocc])
        bd = fermitools.scf.density(bc[:, :nbocc])

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

    # Integrals
    s = interface.integrals.overlap(basis, labels, coords)
    h = interface.integrals.core_hamiltonian(basis, labels, coords)
    r = interface.integrals.repulsion(basis, labels, coords)

    # Call the solver
    en_elec, c = solve(na, nb, s, h, r, e_thresh=1e-14)

    return en_elec if not return_coeffs else (en_elec, c)


def perturbed_energy_function(basis, labels, coords, charge, spin,
                              niter=50, e_thresh=1e-12):
    # Spaces
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)

    # Integrals
    s = interface.integrals.overlap(basis, labels, coords)
    p = interface.integrals.dipole(basis, labels, coords)
    h = interface.integrals.core_hamiltonian(basis, labels, coords)
    r = interface.integrals.repulsion(basis, labels, coords)

    def electronic_energy(f=(0., 0., 0.)):
        h_pert = h - numpy.tensordot(f, p, axes=(0, 0))
        en_elec, (ac, bc) = solve(na, nb, s, h_pert, r, e_thresh=e_thresh)
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
    p = interface.integrals.dipole(BASIS, LABELS, COORDS)
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    ad = fermitools.scf.density(ac[:, :na])
    bd = fermitools.scf.density(bc[:, :nb])
    mu = numpy.array([numpy.vdot(px, ad) + numpy.vdot(px, bd) for px in p])

    # Evaluate dipole moment as energy derivative
    en_f = perturbed_energy_function(BASIS, LABELS, COORDS, CHARGE, SPIN,
                                     e_thresh=1e-15)
    en_df = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                               step=0.0025, npts=13)

    print(en_df.round(11))
    print(mu.round(11))

    # Tests
    from numpy.testing import assert_almost_equal
    assert_almost_equal(en_tot, -74.66178436045595, decimal=10)

    # This can be converged more tightly by including the orbital gradient
    # as a convergence threshold, I think
    assert_almost_equal(en_df, -mu, decimal=8)


if __name__ == '__main__':
    main()
