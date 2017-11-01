def test__rmp2():
    import scripts.rmp2
    scripts.rmp2.main()


def test__ump2():
    import scripts.ump2
    scripts.ump2.main()


def test__mp2():
    import scripts.mp2
    scripts.mp2.main()


def test__cepa0():
    import scripts.cepa0
    scripts.cepa0.main()


def test__scf():
    import scripts.scf
    scripts.scf.main()


def test__omp2():
    import scripts.omp2
    scripts.omp2.main()


def test__ocepa0():
    import scripts.ocepa0
    scripts.ocepa0.main()


def test__odc12():
    import scripts.odc12
    scripts.odc12.main()


def test__uhf():
    import scripts.uhf
    scripts.uhf.main()


def test__rpa():
    import scripts.rpa
    scripts.rpa.main()


def test__lr_scf():
    import scripts.lr_scf as lr
    import scripts.scf as scf

    import numpy
    import scipy.linalg as spla

    import fermitools
    import interfaces.psi4 as interface

    CHARGE = 0
    SPIN = 0
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nocc = na + nb

    # Integrals
    nbf = interface.integrals.nbf(BASIS, LABELS)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # Orbitals
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = spla.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_elec, c = scf.solve_scf(norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso,
                               c_guess=c, niter=200, e_thresh=1e-14,
                               r_thresh=1e-12, print_conv=200)

    # Evaluate the excitation energies by linear response theory
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1 = scf.singles_density(norb=norb, nocc=nocc)
    m2 = scf.doubles_density(m1)
    a_orb = lr.diagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    b_orb = lr.offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    w_rpa = lr.spectrum(a_orb, b_orb)

    # Compare to RPA energies posted on Crawdad
    from numpy.testing import assert_almost_equal
    w_rpa_ref = [0.2851637170, 0.2851637170, 0.2851637170, 0.2997434467,
                 0.2997434467, 0.2997434467, 0.3526266606, 0.3526266606,
                 0.3526266606, 0.3547782530, 0.3651313107, 0.3651313107,
                 0.3651313107, 0.4153174946, 0.5001011401, 0.5106610509,
                 0.5106610509, 0.5106610509, 0.5460719086, 0.5460719086,
                 0.5460719086, 0.5513718846, 0.6502707118, 0.8734253708,
                 1.1038187957, 1.1038187957, 1.1038187957, 1.1957870714,
                 1.1957870714, 1.1957870714, 1.2832053178, 1.3237421886,
                 19.9585040647, 19.9585040647, 19.9585040647, 20.0109471551,
                 20.0113074586, 20.0113074586, 20.0113074586, 20.0504919449]
    assert_almost_equal(w_rpa, w_rpa_ref, decimal=10)

    # Test derivatives
    import os
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    en_dxdx = numpy.load(os.path.join(data_path, 'lr_scf/en_dxdx.npy'))
    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    o = slice(None, nocc)
    v = slice(nocc, None)
    t_orb = numpy.transpose([
        lr.orbital_property_gradient(o, v, px, m1) for px in p])
    r_orb = lr.static_response_vector(a_orb, b_orb, t_orb)
    alpha = lr.static_linear_response_function(t_orb, r_orb)

    # Evaluate dipole polarizability as energy derivative
    en_f_func = scf.perturbed_energy_function(norb=norb, nocc=nocc,
                                              h_aso=h_aso, p_aso=p_aso,
                                              g_aso=g_aso, c_guess=c,
                                              niter=200, e_thresh=1e-14,
                                              r_thresh=1e-12, print_conv=True)
    en_df2 = fermitools.math.central_difference(en_f_func, [0., 0., 0.],
                                                step=0.01, nder=2, npts=9)

    # Compare the two
    assert_almost_equal(numpy.diag(alpha), -en_df2, decimal=9)


def test__lr_ocepa0_neutral():
    import scripts.lr_ocepa0 as lr
    import scripts.ocepa0 as ocepa0

    import numpy
    import scipy.linalg as spla

    import fermitools
    import interfaces.psi4 as interface

    CHARGE = 0
    SPIN = 0
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nocc = na + nb

    # Integrals
    nbf = interface.integrals.nbf(BASIS, LABELS)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # Orbitals
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = spla.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = ocepa0.solve_ocepa0(norb=norb, nocc=nocc, h_aso=h_aso,
                                         g_aso=g_aso, c_guess=c,
                                         t2_guess=t2_guess, niter=200,
                                         e_thresh=1e-14, r_thresh=1e-13,
                                         print_conv=True)

    # Build the diagonal orbital and amplitude Hessian
    o = slice(None, nocc)
    v = slice(nocc, None)
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    f = ocepa0.fock(h[o, o], h[o, v], h[v, v], g[o, o, o, o], g[o, o, o, v],
                    g[o, v, o, v])
    m1_ref = ocepa0.singles_reference_density(norb=norb, nocc=nocc)
    m1_cor = ocepa0.singles_correlation_density(t2)
    m1 = m1_ref + m1_cor
    k2 = ocepa0.doubles_cumulant(t2)
    m2 = ocepa0.doubles_density(m1_ref, m1_cor, k2)

    a_orb = lr.diagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    a_mix = lr.diagonal_mixed_hessian(o, v, g, f, t2)
    a_amp = lr.diagonal_amplitude_hessian(f[o, o], f[v, v], g[o, o, o, o],
                                          g[o, v, o, v], g[v, v, v, v])

    b_orb = lr.offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    b_mix = lr.offdiagonal_mixed_hessian(o, v, g, f, t2)
    b_amp = lr.numpy.zeros_like(a_amp)

    # Test the orbital and amplitude Hessians
    import os
    from numpy.testing import assert_almost_equal

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    en_dxdx = numpy.load(os.path.join(data_path,
                                      'lr_ocepa0/neutral/en_dxdx.npy'))
    en_dtdx = numpy.load(os.path.join(data_path,
                                      'lr_ocepa0/neutral/en_dtdx.npy'))
    en_dxdt = numpy.load(os.path.join(data_path,
                                      'lr_ocepa0/neutral/en_dxdt.npy'))
    en_dtdt = numpy.load(os.path.join(data_path,
                                      'lr_ocepa0/neutral/en_dtdt.npy'))

    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)
    assert_almost_equal(en_dtdx, -2*(a_mix + b_mix), decimal=9)
    assert_almost_equal(en_dxdt, numpy.transpose(en_dtdx), decimal=9)
    assert_almost_equal(en_dtdt, 2*(a_amp + b_amp), decimal=9)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t_orb = numpy.transpose([
        lr.orbital_property_gradient(o, v, px, m1) for px in p])
    t_amp = numpy.transpose([
        lr.amplitude_property_gradient(px[o, o], px[v, v], t2) for px in p])

    a = numpy.bmat([[a_orb, -a_mix.T], [-a_mix, a_amp]])
    b = numpy.bmat([[b_orb, -b_mix.T], [-b_mix, b_amp]])
    t = numpy.bmat([[t_orb], [t_amp]])
    r = lr.static_response_vector(a, b, t)
    alpha = lr.static_linear_response_function(t, r)

    # Evaluate dipole polarizability as energy derivative
    en_f_func = ocepa0.perturbed_energy_function(norb=norb, nocc=nocc,
                                                 h_aso=h_aso, p_aso=p_aso,
                                                 g_aso=g_aso, c_guess=c,
                                                 t2_guess=t2, niter=200,
                                                 e_thresh=1e-14,
                                                 r_thresh=1e-12,
                                                 print_conv=True)
    en_df2 = fermitools.math.central_difference(en_f_func, [0., 0., 0.],
                                                step=0.01, nder=2, npts=9)

    # Compare the two
    assert_almost_equal(numpy.diag(alpha), -en_df2, decimal=9)
