import numpy
import fermitools

from numpy.testing import assert_almost_equal


def en_f_function(h_ao, p_ao, r_ao, co_guess, cv_guess, t2_guess, niter=200,
                  diis_start=3, diis_nvec=20, rthresh=1e-9):

    def _en(f):
        hp_ao = h_ao - numpy.tensordot(f, p_ao, axes=(0, 0))
        en_elec, co, cv, t2, info = fermitools.oo.odc12.solve(
                h_ao=hp_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
                t2_guess=t2_guess, niter=niter, rthresh=rthresh,
                diis_start=diis_start, diis_nvec=diis_nvec, print_conv=False)
        print(info)
        return en_elec

    return _en


def test__main():
    import drivers.odc12
    import interfaces.psi4 as interface

    charge = +1
    spin = 1
    labels = ('H', 'F')
    coords = ((0., 0., 0.),
              (0., 0., 1.))
    basis = 'sto-3g'

    alpha, info = drivers.odc12.polarizability(
            labels=labels,
            coords=coords,
            charge=charge,
            spin=spin,
            basis=basis,
            angstrom=False,
            nvec=100,               # max number of subspace vectors per root
            niter=50,               # number of iterations
            rthresh=1e-6,           # convergence threshold
            oo_niter=200,           # number of iterations for ground state
            oo_rthresh=1e-10,       # convergence threshold for ground state
            diis_start=3,           # when to start DIIS extrapolations
            diis_nvec=20,           # maximum number of DIIS vectors
            interface=interface)    # interface for computing integrals

    mu = info['mu']

    # Integrals
    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    p_ao = interface.integrals.dipole(basis, labels, coords)
    r_ao = interface.integrals.repulsion(basis, labels, coords)

    # Differentiate
    en_f_ = en_f_function(
            h_ao=h_ao, p_ao=p_ao, r_ao=r_ao, co_guess=info['co'],
            cv_guess=info['cv'], t2_guess=info['t2'], niter=200, rthresh=1e-11)
    en_elec = en_f_((0., 0., 0.))
    print(en_elec)

    print("First derivative")
    en_df = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.03, nder=1, npts=15)
    print(en_df)

    print("Second derivative")
    en_df2 = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.03, nder=2, npts=15)
    print(en_df2)

    print("Compare dE/df to <Psi|mu|Psi>:")
    print(en_df.round(10))
    print(mu.round(10))
    print(max(numpy.abs(en_df + mu)))

    print("Compare d2E/df2 to <<mu; mu>>:")
    print(en_df2.round(10))
    print(alpha.round(10))
    print(max(numpy.abs(en_df2 - numpy.diag(alpha))))

    assert_almost_equal(en_df, -mu, decimal=9)
    assert_almost_equal(en_df2, numpy.diag(alpha), decimal=9)


if __name__ == '__main__':
    test__main()
