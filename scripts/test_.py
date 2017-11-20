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
    import scipy.linalg

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
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_elec, c = scf.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            niter=200, e_thresh=1e-14, r_thresh=1e-12, print_conv=200)

    # Evaluate the excitation energies by linear response theory
    o = slice(None, nocc)
    v = slice(nocc, None)
    no = nocc
    nv = norb - nocc
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1oo = numpy.eye(no)
    m1vv = numpy.zeros((nv, nv))
    m2oooo = (+ numpy.einsum('ik,jl->ijkl', m1oo, m1oo)
              - numpy.einsum('il,jk->ijkl', m1oo, m1oo))
    m2oovv = numpy.zeros((no, no, nv, nv))
    m2ovov = numpy.zeros((no, nv, no, nv))
    m2vvvv = numpy.zeros((nv, nv, nv, nv))

    v_raveler = fermitools.math.raveler({0: (0, 1)})
    m_raveler = fermitools.math.raveler({0: (0, 1), 1: (2, 3)})

    a = m_raveler(lr.orbital_hessian_diag(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1oo, m1vv, m2oooo, m2oovv,
            m2ovov, m2vvvv))
    b = m_raveler(lr.orbital_hessian_offd(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2oooo, m2oovv, m2ovov, m2vvvv))
    w_rpa = lr.spectrum(a, b)

    # Compare to RPA energies posted on Crawdad
    from numpy.testing import assert_almost_equal
    w_rpa_ref = [
            0.2851637170, 0.2851637170, 0.2851637170, 0.2997434467,
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
    data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
    en_dxdx = numpy.load(os.path.join(data_path, 'lr_scf/en_dxdx.npy'))
    assert_almost_equal(en_dxdx, 2*(a + b), decimal=9)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t = v_raveler(lr.orbital_property_gradient(p[:, o, v], m1oo, m1vv))
    r = lr.static_response_vector(a, b, t)
    alpha = lr.static_linear_response_function(t, r)

    # Evaluate dipole polarizability as energy derivative
    en_f_func = scf.perturbed_energy_function(
            norb=norb, nocc=nocc, h_aso=h_aso, p_aso=p_aso, g_aso=g_aso,
            c_guess=c, niter=200, e_thresh=1e-14, r_thresh=1e-12,
            print_conv=True)
    en_df2 = fermitools.math.central_difference(
            en_f_func, [0., 0., 0.], step=0.02, nder=2, npts=15)

    # Compare the two
    assert_almost_equal(numpy.diag(alpha), en_df2, decimal=9)


def test__lr_ocepa0_cation():
    import scripts.lr_ocepa0 as lr
    import scripts.ocepa0 as ocepa0

    import numpy
    import scipy.linalg

    import fermitools
    import interfaces.psi4 as interface

    CHARGE = +1
    SPIN = 1
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
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = ocepa0.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
            print_conv=True)

    # Build the diagonal orbital and amplitude Hessian
    no = nocc
    o = slice(None, nocc)
    v = slice(nocc, None)
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    dm1oo = numpy.eye(no)
    cm1oo, cm1vv = fermitools.oo.ocepa0.onebody_correlation_density(t2)
    m1oo = dm1oo + cm1oo
    m1vv = cm1vv

    k2oooo = fermitools.oo.ocepa0.twobody_cumulant_oooo(t2)
    k2oovv = fermitools.oo.ocepa0.twobody_cumulant_oovv(t2)
    k2ovov = fermitools.oo.ocepa0.twobody_cumulant_ovov(t2)
    k2vvvv = fermitools.oo.ocepa0.twobody_cumulant_vvvv(t2)

    m2oooo = fermitools.oo.ocepa0.twobody_moment_oooo(dm1oo, cm1oo, k2oooo)
    m2oovv = fermitools.oo.ocepa0.twobody_moment_oovv(k2oovv)
    m2ovov = fermitools.oo.ocepa0.twobody_moment_ovov(dm1oo, cm1vv, k2ovov)
    m2vvvv = fermitools.oo.ocepa0.twobody_moment_vvvv(k2vvvv)

    foo = fermitools.oo.ocepa0.fock_oo(h[o, o], g[o, o, o, o])
    fov = fermitools.oo.ocepa0.fock_oo(h[o, v], g[o, o, o, v])
    fvv = fermitools.oo.ocepa0.fock_vv(h[v, v], g[o, v, o, v])

    v_orb_raveler = fermitools.math.raveler({0: (0, 1)})
    v_amp_raveler = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    m_orb_raveler = fermitools.math.raveler({0: (0, 1), 1: (2, 3)})
    m_mix_raveler = fermitools.math.asym.megaraveler(
            {0: ((0,), (1,)), 1: ((2, 3), (4, 5))})
    m_amp_raveler = fermitools.math.asym.megaraveler(
            {0: ((0, 1), (2, 3)), 1: ((4, 5), (6, 7))})

    a_orb = m_orb_raveler(lr.orbital_hessian_diag(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1oo, m1vv, m2oooo, m2oovv,
            m2ovov, m2vvvv))
    a_mix = m_mix_raveler(lr.mixed_hessian_diag(
            fov, g[o, o, o, v], g[o, v, v, v], t2))
    a_amp = m_amp_raveler(lr.amplitude_hessian(
            foo, fvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v]))

    b_orb = m_orb_raveler(lr.orbital_hessian_offd(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2oooo, m2oovv, m2ovov, m2vvvv))
    b_mix = m_mix_raveler(lr.mixed_hessian_offd(
            fov, g[o, o, o, v], g[o, v, v, v], t2))
    b_amp = lr.numpy.zeros_like(a_amp)

    # Test the orbital and amplitude Hessians
    import os
    from numpy.testing import assert_almost_equal

    data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
    en_dxdx = numpy.load(
            os.path.join(data_path, 'lr_ocepa0/cation/en_dxdx.npy'))
    en_dtdx = numpy.load(
            os.path.join(data_path, 'lr_ocepa0/cation/en_dtdx.npy'))
    en_dxdt = numpy.load(
            os.path.join(data_path, 'lr_ocepa0/cation/en_dxdt.npy'))
    en_dtdt = numpy.load(
            os.path.join(data_path, 'lr_ocepa0/cation/en_dtdt.npy'))

    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)
    assert_almost_equal(en_dxdt, 2*(a_mix + b_mix), decimal=9)
    assert_almost_equal(en_dxdt, numpy.transpose(en_dtdx), decimal=9)
    assert_almost_equal(en_dtdt, 2*(a_amp + b_amp), decimal=9)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t_orb = v_orb_raveler(lr.orbital_property_gradient(
            p[:, o, v], m1oo, m1vv))
    t_amp = v_amp_raveler(lr.amplitude_property_gradient(
            p[:, o, o], p[:, v, v], t2))

    a = numpy.bmat([[a_orb, a_mix], [a_mix.T, a_amp]])
    b = numpy.bmat([[b_orb, b_mix], [b_mix.T, b_amp]])
    t = numpy.bmat([[t_orb], [t_amp]])
    r = lr.static_response_vector(a, b, t)
    alpha = lr.static_linear_response_function(t, r)

    # Evaluate dipole polarizability as energy derivative
    en_f_func = ocepa0.perturbed_energy_function(
            norb=norb, nocc=nocc, h_aso=h_aso, p_aso=p_aso, g_aso=g_aso,
            c_guess=c, t2_guess=t2, niter=200, e_thresh=1e-14, r_thresh=1e-12,
            print_conv=True)
    en_df2 = fermitools.math.central_difference(
            en_f_func, [0., 0., 0.], step=0.02, nder=2, npts=15)

    # Compare the two
    assert_almost_equal(numpy.diag(alpha), en_df2, decimal=9)


def test__lr_ocepa0_neutral():
    import scripts.lr_ocepa0 as lr
    import scripts.ocepa0 as ocepa0

    import numpy
    import scipy.linalg

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
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = ocepa0.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
            print_conv=True)

    # Build the diagonal orbital and amplitude Hessian
    no = nocc
    o = slice(None, nocc)
    v = slice(nocc, None)
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    dm1oo = numpy.eye(no)
    cm1oo, cm1vv = fermitools.oo.ocepa0.onebody_correlation_density(t2)
    m1oo = dm1oo + cm1oo
    m1vv = cm1vv

    k2oooo = fermitools.oo.ocepa0.twobody_cumulant_oooo(t2)
    k2oovv = fermitools.oo.ocepa0.twobody_cumulant_oovv(t2)
    k2ovov = fermitools.oo.ocepa0.twobody_cumulant_ovov(t2)
    k2vvvv = fermitools.oo.ocepa0.twobody_cumulant_vvvv(t2)

    m2oooo = fermitools.oo.ocepa0.twobody_moment_oooo(dm1oo, cm1oo, k2oooo)
    m2oovv = fermitools.oo.ocepa0.twobody_moment_oovv(k2oovv)
    m2ovov = fermitools.oo.ocepa0.twobody_moment_ovov(dm1oo, cm1vv, k2ovov)
    m2vvvv = fermitools.oo.ocepa0.twobody_moment_vvvv(k2vvvv)

    foo = fermitools.oo.ocepa0.fock_oo(h[o, o], g[o, o, o, o])
    fov = fermitools.oo.ocepa0.fock_oo(h[o, v], g[o, o, o, v])
    fvv = fermitools.oo.ocepa0.fock_vv(h[v, v], g[o, v, o, v])

    v_orb_raveler = fermitools.math.raveler({0: (0, 1)})
    v_amp_raveler = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    m_orb_raveler = fermitools.math.raveler({0: (0, 1), 1: (2, 3)})
    m_mix_raveler = fermitools.math.asym.megaraveler(
            {0: ((0,), (1,)), 1: ((2, 3), (4, 5))})
    m_amp_raveler = fermitools.math.asym.megaraveler(
            {0: ((0, 1), (2, 3)), 1: ((4, 5), (6, 7))})

    a_orb = m_orb_raveler(lr.orbital_hessian_diag(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1oo, m1vv, m2oooo, m2oovv,
            m2ovov, m2vvvv))
    a_mix = m_mix_raveler(lr.mixed_hessian_diag(
            fov, g[o, o, o, v], g[o, v, v, v], t2))
    a_amp = m_amp_raveler(lr.amplitude_hessian(
            foo, fvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v]))

    b_orb = m_orb_raveler(lr.orbital_hessian_offd(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2oooo, m2oovv, m2ovov, m2vvvv))
    b_mix = m_mix_raveler(lr.mixed_hessian_offd(
            fov, g[o, o, o, v], g[o, v, v, v], t2))
    b_amp = lr.numpy.zeros_like(a_amp)

    # Test the orbital and amplitude Hessians
    import os
    from numpy.testing import assert_almost_equal

    data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
    en_dxdx = numpy.load(
            os.path.join(data_path, 'lr_ocepa0/neutral/en_dxdx.npy'))
    en_dtdx = numpy.load(
            os.path.join(data_path, 'lr_ocepa0/neutral/en_dtdx.npy'))
    en_dxdt = numpy.load(
            os.path.join(data_path, 'lr_ocepa0/neutral/en_dxdt.npy'))
    en_dtdt = numpy.load(
            os.path.join(data_path, 'lr_ocepa0/neutral/en_dtdt.npy'))

    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)
    assert_almost_equal(en_dxdt, 2*(a_mix + b_mix), decimal=9)
    assert_almost_equal(en_dxdt, numpy.transpose(en_dtdx), decimal=9)
    assert_almost_equal(en_dtdt, 2*(a_amp + b_amp), decimal=9)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t_orb = v_orb_raveler(lr.orbital_property_gradient(
            p[:, o, v], m1oo, m1vv))
    t_amp = v_amp_raveler(lr.amplitude_property_gradient(
            p[:, o, o], p[:, v, v], t2))

    a = numpy.bmat([[a_orb, a_mix], [a_mix.T, a_amp]])
    b = numpy.bmat([[b_orb, b_mix], [b_mix.T, b_amp]])
    t = numpy.bmat([[t_orb], [t_amp]])
    r = lr.static_response_vector(a, b, t)
    alpha = lr.static_linear_response_function(t, r)

    # Evaluate dipole polarizability as energy derivative
    en_f_func = ocepa0.perturbed_energy_function(
            norb=norb, nocc=nocc, h_aso=h_aso, p_aso=p_aso, g_aso=g_aso,
            c_guess=c, t2_guess=t2, niter=200, e_thresh=1e-14, r_thresh=1e-12,
            print_conv=True)
    en_df2 = fermitools.math.central_difference(
            en_f_func, [0., 0., 0.], step=0.02, nder=2, npts=15)

    # Compare the two
    assert_almost_equal(numpy.diag(alpha), en_df2, decimal=9)


def test__lr_odc12_cation():
    import scripts.lr_odc12 as lr
    import scripts.odc12 as odc12

    import numpy
    import scipy.linalg

    import fermitools
    import interfaces.psi4 as interface

    CHARGE = +1
    SPIN = 1
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
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve ODC12
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = odc12.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
            print_conv=True)

    # Build the diagonal orbital and amplitude Hessian
    no = nocc
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    dm1oo = numpy.eye(no)
    cm1oo, m1vv = fermitools.oo.odc12.onebody_correlation_density(t2)
    m1oo = dm1oo + cm1oo
    k2oooo = fermitools.oo.odc12.twobody_cumulant_oooo(t2)
    k2oovv = fermitools.oo.odc12.twobody_cumulant_oovv(t2)
    k2ovov = fermitools.oo.odc12.twobody_cumulant_ovov(t2)
    k2vvvv = fermitools.oo.odc12.twobody_cumulant_vvvv(t2)
    m2oooo = fermitools.oo.odc12.twobody_moment_oooo(m1oo, k2oooo)
    m2oovv = fermitools.oo.odc12.twobody_moment_oovv(k2oovv)
    m2ovov = fermitools.oo.odc12.twobody_moment_ovov(m1oo, m1vv, k2ovov)
    m2vvvv = fermitools.oo.odc12.twobody_moment_vvvv(m1vv, k2vvvv)

    o = slice(None, nocc)
    v = slice(nocc, None)
    foo = fermitools.oo.odc12.fock_oo(
            h[o, o], g[o, o, o, o], g[o, v, o, v], m1oo, m1vv)
    fov = fermitools.oo.odc12.fock_oo(
            h[o, v], g[o, o, o, v], g[o, v, v, v], m1oo, m1vv)
    fvv = fermitools.oo.odc12.fock_vv(
            h[v, v], g[o, v, o, v], g[v, v, v, v], m1oo, m1vv)
    foo = fermitools.oo.odc12.fock_oo(
            h[o, o], g[o, o, o, o], g[o, v, o, v], m1oo, m1vv)
    fov = fermitools.oo.odc12.fock_oo(
            h[o, v], g[o, o, o, v], g[o, v, v, v], m1oo, m1vv)
    fvv = fermitools.oo.odc12.fock_vv(
            h[v, v], g[o, v, o, v], g[v, v, v, v], m1oo, m1vv)
    ffoo = fermitools.oo.odc12.fancy_property(foo, m1oo)
    ffvv = fermitools.oo.odc12.fancy_property(fvv, m1vv)
    fg = lr.fancy_repulsion(
            ffoo, ffvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v],
            m1oo, m1vv)
    fi = lr.fancy_mixed_interaction(
            fov, g[o, o, o, v], g[o, v, v, v], m1oo, m1vv)

    # Raveling operators
    v_orb_raveler = fermitools.math.raveler({0: (0, 1)})
    v_amp_raveler = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    m_orb_raveler = fermitools.math.raveler({0: (0, 1), 1: (2, 3)})
    m_mix_raveler = fermitools.math.asym.megaraveler(
            {0: ((0,), (1,)), 1: ((2, 3), (4, 5))})
    m_amp_raveler = fermitools.math.asym.megaraveler(
            {0: ((0, 1), (2, 3)), 1: ((4, 5), (6, 7))})

    a_orb = m_orb_raveler(lr.orbital_hessian_diag(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1oo, m1vv, m2oooo, m2oovv,
            m2ovov, m2vvvv))
    a_mix = m_mix_raveler(lr.mixed_hessian_diag(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2))
    a_amp = m_amp_raveler(lr.amplitude_hessian_diag(
            ffoo, ffvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v],
            fg['o,o,o,o'], fg['o,v,o,v'], fg['v,v,v,v'], t2))
    b_orb = m_orb_raveler(lr.orbital_hessian_offd(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2oooo, m2oovv, m2ovov, m2vvvv))
    b_mix = m_mix_raveler(lr.mixed_hessian_offd(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2))
    b_amp = m_amp_raveler(lr.amplitude_hessian_offd(
            fg['o,o,o,o'], fg['o,v,o,v'], fg['v,v,v,v'], t2))

    # Test the orbital and amplitude Hessians
    import os
    from numpy.testing import assert_almost_equal

    data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
    en_dxdx = numpy.real(
            numpy.load(os.path.join(data_path, 'lr_odc12/cation/en_dxdx.npy')))
    en_dxdt = numpy.real(
            numpy.load(os.path.join(data_path, 'lr_odc12/cation/en_dxdt.npy')))
    en_dtdx = numpy.real(
            numpy.load(os.path.join(data_path, 'lr_odc12/cation/en_dtdx.npy')))
    en_dtdt = numpy.real(
            numpy.load(os.path.join(data_path, 'lr_odc12/cation/en_dtdt.npy')))

    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)
    assert_almost_equal(en_dxdt, 2*(a_mix + b_mix), decimal=9)
    assert_almost_equal(en_dxdt, numpy.transpose(en_dtdx), decimal=9)
    assert_almost_equal(en_dtdt, 2*(a_amp + b_amp), decimal=8)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    fpoo = fermitools.oo.odc12.fancy_property(p[:, o, o], m1oo)
    fpvv = fermitools.oo.odc12.fancy_property(p[:, v, v], m1vv)
    t_orb = v_orb_raveler(lr.orbital_property_gradient(
            p[:, o, v], m1oo, m1vv))
    t_amp = v_amp_raveler(lr.amplitude_property_gradient(fpoo, -fpvv, t2))

    a = numpy.bmat([[a_orb, a_mix], [a_mix.T, a_amp]])
    b = numpy.bmat([[b_orb, b_mix], [b_mix.T, b_amp]])
    t = numpy.bmat([[t_orb], [t_amp]])
    r = lr.static_response_vector(a, b, t)
    alpha = lr.static_linear_response_function(t, r)

    # Evaluate dipole polarizability as energy derivative
    en_f_func = odc12.perturbed_energy_function(
            norb=norb, nocc=nocc, h_aso=h_aso, p_aso=p_aso, g_aso=g_aso,
            c_guess=c, t2_guess=t2, niter=200, e_thresh=1e-14, r_thresh=1e-12,
            print_conv=True)
    en_df2 = fermitools.math.central_difference(
            en_f_func, [0., 0., 0.], step=0.02, nder=2, npts=15)

    # Compare the two
    assert_almost_equal(numpy.diag(alpha), en_df2, decimal=9)


def test__lr_odc12_neutral():
    import scripts.lr_odc12 as lr
    import scripts.odc12 as odc12

    import numpy
    import scipy.linalg

    import fermitools
    import interfaces.psi4 as interface

    CHARGE = +0
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
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = odc12.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
            print_conv=True)

    # Build the diagonal orbital and amplitude Hessian
    no = nocc
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    dm1oo = numpy.eye(no)
    cm1oo, m1vv = fermitools.oo.odc12.onebody_correlation_density(t2)
    m1oo = dm1oo + cm1oo
    k2oooo = fermitools.oo.odc12.twobody_cumulant_oooo(t2)
    k2oovv = fermitools.oo.odc12.twobody_cumulant_oovv(t2)
    k2ovov = fermitools.oo.odc12.twobody_cumulant_ovov(t2)
    k2vvvv = fermitools.oo.odc12.twobody_cumulant_vvvv(t2)
    m2oooo = fermitools.oo.odc12.twobody_moment_oooo(m1oo, k2oooo)
    m2oovv = fermitools.oo.odc12.twobody_moment_oovv(k2oovv)
    m2ovov = fermitools.oo.odc12.twobody_moment_ovov(m1oo, m1vv, k2ovov)
    m2vvvv = fermitools.oo.odc12.twobody_moment_vvvv(m1vv, k2vvvv)

    o = slice(None, nocc)
    v = slice(nocc, None)
    foo = fermitools.oo.odc12.fock_oo(
            h[o, o], g[o, o, o, o], g[o, v, o, v], m1oo, m1vv)
    fov = fermitools.oo.odc12.fock_oo(
            h[o, v], g[o, o, o, v], g[o, v, v, v], m1oo, m1vv)
    fvv = fermitools.oo.odc12.fock_vv(
            h[v, v], g[o, v, o, v], g[v, v, v, v], m1oo, m1vv)
    ffoo = fermitools.oo.odc12.fancy_property(foo, m1oo)
    ffvv = fermitools.oo.odc12.fancy_property(fvv, m1vv)
    fg = lr.fancy_repulsion(
            ffoo, ffvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v],
            m1oo, m1vv)
    fi = lr.fancy_mixed_interaction(
            fov, g[o, o, o, v], g[o, v, v, v], m1oo, m1vv)

    # Raveling operators
    v_orb_raveler = fermitools.math.raveler({0: (0, 1)})
    v_amp_raveler = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    m_orb_raveler = fermitools.math.raveler({0: (0, 1), 1: (2, 3)})
    m_mix_raveler = fermitools.math.asym.megaraveler(
            {0: ((0,), (1,)), 1: ((2, 3), (4, 5))})
    m_amp_raveler = fermitools.math.asym.megaraveler(
            {0: ((0, 1), (2, 3)), 1: ((4, 5), (6, 7))})

    a_orb = m_orb_raveler(lr.orbital_hessian_diag(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1oo, m1vv, m2oooo, m2oovv,
            m2ovov, m2vvvv))
    a_mix = m_mix_raveler(lr.mixed_hessian_diag(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2))
    a_amp = m_amp_raveler(lr.amplitude_hessian_diag(
            ffoo, ffvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v],
            fg['o,o,o,o'], fg['o,v,o,v'], fg['v,v,v,v'], t2))
    b_orb = m_orb_raveler(lr.orbital_hessian_offd(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2oooo, m2oovv, m2ovov, m2vvvv))
    b_mix = m_mix_raveler(lr.mixed_hessian_offd(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2))
    b_amp = m_amp_raveler(lr.amplitude_hessian_offd(
            fg['o,o,o,o'], fg['o,v,o,v'], fg['v,v,v,v'], t2))

    # Test the orbital and amplitude Hessians
    import os
    from numpy.testing import assert_almost_equal

    data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
    en_dxdx = numpy.real(numpy.load(
            os.path.join(data_path, 'lr_odc12/neutral/en_dxdx.npy')))
    en_dxdt = numpy.real(numpy.load(
            os.path.join(data_path, 'lr_odc12/neutral/en_dxdt.npy')))
    en_dtdx = numpy.real(numpy.load(
            os.path.join(data_path, 'lr_odc12/neutral/en_dtdx.npy')))
    en_dtdt = numpy.real(numpy.load(
            os.path.join(data_path, 'lr_odc12/neutral/en_dtdt.npy')))

    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)
    assert_almost_equal(en_dxdt, 2*(a_mix + b_mix), decimal=9)
    assert_almost_equal(en_dxdt, numpy.transpose(en_dtdx), decimal=9)
    assert_almost_equal(en_dtdt, 2*(a_amp + b_amp), decimal=8)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    fpoo = fermitools.oo.odc12.fancy_property(p[:, o, o], m1oo)
    fpvv = fermitools.oo.odc12.fancy_property(p[:, v, v], m1vv)
    t_orb = v_orb_raveler(lr.orbital_property_gradient(
            p[:, o, v], m1oo, m1vv))
    t_amp = v_amp_raveler(lr.amplitude_property_gradient(fpoo, -fpvv, t2))

    a = numpy.bmat([[a_orb, a_mix], [a_mix.T, a_amp]])
    b = numpy.bmat([[b_orb, b_mix], [b_mix.T, b_amp]])
    t = numpy.bmat([[t_orb], [t_amp]])
    r = lr.static_response_vector(a, b, t)
    alpha = lr.static_linear_response_function(t, r)

    # Evaluate dipole polarizability as energy derivative
    en_f_func = odc12.perturbed_energy_function(
            norb=norb, nocc=nocc, h_aso=h_aso, p_aso=p_aso, g_aso=g_aso,
            c_guess=c, t2_guess=t2, niter=200, e_thresh=1e-14, r_thresh=1e-12,
            print_conv=True)
    en_df2 = fermitools.math.central_difference(
            en_f_func, [0., 0., 0.], step=0.02, nder=2, npts=15)

    # Compare the two
    assert_almost_equal(numpy.diag(alpha), en_df2, decimal=9)
