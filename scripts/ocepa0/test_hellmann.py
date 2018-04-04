import numpy
import fermitools

from numpy.testing import assert_almost_equal


def en_f_function(h_ao, p_ao, r_ao, co_guess, cv_guess, t2_guess, maxiter=200,
                  diis_start=3, diis_nvec=20, rthresh=1e-9):

    def _en(f):
        hp_ao = h_ao - numpy.tensordot(f, p_ao, axes=(0, 0))
        en_elec, co, cv, t2, info = fermitools.oo.ocepa0.solve(
                h_ao=hp_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
                t2_guess=t2_guess, maxiter=maxiter, rthresh=rthresh,
                diis_start=diis_start, diis_nvec=diis_nvec, print_conv=False)
        print(info)
        return en_elec

    return _en


def test_main():
    import drivers
    import interfaces.psi4 as interface

    charge = +1
    spin = 1
    labels = ('H', 'F')
    coords = ((0., 0., 0.),
              (0., 0., 1.))
    basis = 'sto-3g'
    oo_maxiter = 200
    oo_rthresh = 1e-10
    diis_start = 3
    diis_nvec = 20
    maxiter = 50
    rthresh = 1e-6

    # Integrals
    h_ao, r_ao, p_ao = drivers.integrals(
            basis, labels, coords, angstrom=False, interface=interface)
    co_guess, cv_guess, no, nv = drivers.hf_orbitals(
            labels, coords, charge, spin, basis, angstrom=False,
            interface=interface)
    t2_guess = numpy.zeros((no, no, nv, nv))

    en_elec, co, cv, t2, info = fermitools.oo.ocepa0.solve(
                h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
                t2_guess=t2_guess, maxiter=oo_maxiter, rthresh=oo_rthresh,
                diis_start=diis_start, diis_nvec=diis_nvec, print_conv=True)
    mu = fermitools.oo.ocepa0.compute_property(p_ao, co, cv, t2)
    alpha = fermitools.lr.ocepa0.solve_static_response(
            h_ao, p_ao, r_ao, co, cv, t2, maxiter=maxiter, rthresh=rthresh,
            print_conv=True)

    # Differentiate
    en_f_ = en_f_function(
            h_ao=h_ao, p_ao=p_ao, r_ao=r_ao, co_guess=co, cv_guess=cv,
            t2_guess=t2, maxiter=200, rthresh=1e-11)
    en_elec = en_f_((0., 0., 0.))
    print(en_elec)

    print("First derivative")
    en_df = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.001, nder=1, npts=5)
    print(en_df)

    print("Second derivative")
    en_df2 = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.010, nder=2, npts=25)
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
    test_main()
