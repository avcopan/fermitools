import numpy
import scipy
import functools
from toolz import functoolz

import fermitools
import interfaces.psi4 as interface
import solvers


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
en_elec, c, t2 = solvers.oo.ocepa0.solve(
        norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
        t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
        print_conv=True)

# Build the diagonal orbital and amplitude Hessian
no = nocc
nv = norb - nocc
nsingles = no * nv
ndoubles = no * (no - 1) * nv * (nv - 1) // 4
o = slice(None, no)
v = slice(no, None)
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


def v_orb_unraveler(r1):
    shape = (no, nv) if r1.ndim == 1 else (no, nv) + r1.shape[1:]
    return numpy.reshape(r1, shape)


def v_amp_unraveler(r2):
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2
    shape = (noo, nvv) if r2.ndim == 1 else (noo, nvv) + r2.shape[1:]
    unravf = fermitools.math.asym.unraveler({0: (0, 1), 1: (2, 3)})
    return unravf(numpy.reshape(r2, shape))


# Evaluate dipole polarizability using linear response theory
p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
p = fermitools.math.transform(p_aso, {1: c, 2: c})
t_d1 = v_orb_raveler(fermitools.lr.ocepa0.t_d1(p[:, o, v], m1oo, m1vv))
t_d2 = v_amp_raveler(fermitools.lr.ocepa0.t_d2(p[:, o, o], p[:, v, v], t2))

t = numpy.concatenate((t_d1, t_d2), axis=0)

a_d1d1_rf = fermitools.lr.ocepa0.a_d1d1_rf(
        h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
        g[v, v, v, v], m1oo, m1vv, m2oooo, m2oovv, m2ovov, m2vvvv)
b_d1d1_rf = fermitools.lr.ocepa0.b_d1d1_rf(
        g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
        m2oooo, m2oovv, m2ovov, m2vvvv)
s_d1d1_rf = fermitools.lr.ocepa0.s_d1d1_rf(m1oo, m1vv)

a_d1d2_rf = fermitools.lr.ocepa0.a_d1d2_rf(
        fov, g[o, o, o, v], g[o, v, v, v], t2)
b_d1d2_rf = fermitools.lr.ocepa0.b_d1d2_rf(
        fov, g[o, o, o, v], g[o, v, v, v], t2)
a_d1d2_lf = fermitools.lr.ocepa0.a_d1d2_lf(
        fov, g[o, o, o, v], g[o, v, v, v], t2)
b_d1d2_lf = fermitools.lr.ocepa0.b_d1d2_lf(
        fov, g[o, o, o, v], g[o, v, v, v], t2)
a_d2d2_rf = fermitools.lr.ocepa0.a_d2d2_rf(
        foo, fvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v])

# Orbital terms
s_d1d1_rf = functoolz.compose(v_orb_raveler, s_d1d1_rf, v_orb_unraveler)
e_sum_d1d1_rf = functoolz.compose(
        v_orb_raveler, fermitools.func.add(a_d1d1_rf, b_d1d1_rf),
        v_orb_unraveler)
e_dif_d1d1_rf = functoolz.compose(
        v_orb_raveler, fermitools.func.sub(a_d1d1_rf, b_d1d1_rf),
        v_orb_unraveler)
# Mixted right terms
e_sum_d1d2_rf = functoolz.compose(
        v_orb_raveler, fermitools.func.add(a_d1d2_rf, b_d1d2_rf),
        v_amp_unraveler)
e_dif_d1d2_rf = functoolz.compose(
        v_orb_raveler, fermitools.func.sub(a_d1d2_rf, b_d1d2_rf),
        v_amp_unraveler)
# Mixed left terms
e_sum_d1d2_lf = functoolz.compose(
        v_amp_raveler, fermitools.func.add(a_d1d2_lf, b_d1d2_lf),
        v_orb_unraveler)
e_dif_d1d2_lf = functoolz.compose(
        v_amp_raveler, fermitools.func.sub(a_d1d2_lf, b_d1d2_lf),
        v_orb_unraveler)
# Amplitude terms
e_d2d2_rf = functoolz.compose(v_amp_raveler, a_d2d2_rf, v_amp_unraveler)

# Combined
e_sum_rf = solvers.lr.ocepa0.e_rf(
        nsingles, e_sum_d1d1_rf, e_sum_d1d2_rf, e_sum_d1d2_lf, e_d2d2_rf)
e_dif_rf = solvers.lr.ocepa0.e_rf(
        nsingles, e_dif_d1d1_rf, e_dif_d1d2_rf, e_dif_d1d2_lf, e_d2d2_rf)

s_d1d1 = s_d1d1_rf(numpy.eye(nsingles))
x_d1d1 = scipy.linalg.inv(s_d1d1)
x_d1d1_rf = scipy.sparse.linalg.aslinearoperator(x_d1d1)
x_rf = solvers.lr.ocepa0.x_rf(nsingles, x_d1d1_rf)

e_eff_rf = solvers.lr.ocepa0.e_eff_rf(e_sum_rf, e_dif_rf, x_rf)


# Response function
n = nsingles + ndoubles
e_sum_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=e_sum_rf)
r_solver_ = functools.partial(scipy.sparse.linalg.cg, e_sum_)
rs, _ = zip(*map(r_solver_, -2 * numpy.moveaxis(t, -1, 0)))
r = numpy.moveaxis(tuple(rs), -1, 0)
alpha = numpy.tensordot(r, t, axes=(0, 0))
print(alpha.round(8))

# Excitation energies
e_eff_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=e_eff_rf)
w2, u = scipy.sparse.linalg.eigs(e_eff_, k=n-2, which='SR')
w = numpy.sqrt(numpy.real(sorted(w2)))
print(w)
