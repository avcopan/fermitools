import numpy
import scipy
import functools

import fermitools
import interfaces.psi4 as interface

from . import scf


CHARGE = 0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

# Nuclear repulsion energy
en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)

# Spaces
na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
nocc = na + nb

# Integrals
nbf = interface.integrals.nbf(BASIS, LABELS)
norb = 2 * nbf
h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

# Orbitals
ac, bc = interface.hf.unrestricted_orbitals(
        BASIS, LABELS, COORDS, CHARGE, SPIN)
c_unsrt = scipy.linalg.block_diag(ac, bc)
sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
c_unsrt = scipy.linalg.block_diag(ac, bc)
c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))
en_elec, c = scf.solve(
        norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
        niter=200, e_thresh=1e-14, r_thresh=1e-12, print_conv=200)
print("Total energy:")
print('{:20.15f}'.format(en_elec + en_nuc))

# Ravel/unraveling operators
v_raveler = fermitools.math.raveler({0: (0, 1)})
m_raveler = fermitools.math.raveler({0: (0, 1), 1: (2, 3)})


def v_unraveler(r1):
    no, nv = nocc, norb - nocc
    shape = (no, nv) if r1.ndim == 1 else (no, nv) + r1.shape[1:]
    return numpy.reshape(r1, shape)


def add(f, g):

    def _sum(*args, **kwargs):
        return f(*args, **kwargs) + g(*args, **kwargs)

    return _sum


def sub(f, g):

    def _diff(*args, **kwargs):
        return f(*args, **kwargs) - g(*args, **kwargs)

    return _diff


# matrices
co = c[:, :nocc]
cv = c[:, nocc:]
hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
pov = fermitools.math.transform(p_aso, {1: co, 2: cv})
goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
foo = fermitools.oo.hf.fock_oo(hoo, goooo)
fvv = fermitools.oo.hf.fock_vv(hvv, govov)
t = v_raveler(fermitools.lr.hf.t_d1(pov))
a_ = fermitools.lr.hf.a_d1d1_left(foo, fvv, govov)
b_ = fermitools.lr.hf.b_d1d1_left(goovv)


def e_sum_(r1):
    a_add_b_ = add(a_, b_)
    return v_raveler(a_add_b_(v_unraveler(r1)))


def e_dif_(r1):
    a_sub_b_ = sub(a_, b_)
    return v_raveler(a_sub_b_(v_unraveler(r1)))


def e_eff_(r1):
    return e_sum_(e_dif_(r1))


# Properties
dim = nocc * (norb - nocc)
e_ = scipy.sparse.linalg.LinearOperator((dim, dim), matvec=e_sum_)
r_solver_ = functools.partial(scipy.sparse.linalg.cg, e_)
rs, _ = zip(*map(r_solver_, -2 * numpy.moveaxis(t, -1, 0)))
r = numpy.moveaxis(tuple(rs), -1, 0)
alpha = numpy.tensordot(r, t, axes=(0, 0))
print(alpha.round(8))
