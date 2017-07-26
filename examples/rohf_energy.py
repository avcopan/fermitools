import numpy
import simplehf
import simplehf.interface.pyscf as interface

CHARGE = +1
SPIN = 1
BASIS = 'STO-3G'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

s = interface.integrals.overlap(BASIS, LABELS, COORDS)
h = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
g = interface.integrals.repulsion(BASIS, LABELS, COORDS)

na = simplehf.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
nb = simplehf.chem.elec.count_beta(LABELS, CHARGE, SPIN)

c_ref = interface.hf.restricted_orbitals(BASIS, LABELS, COORDS,
                                         CHARGE, SPIN)

ad = simplehf.hf.orb.density(na, c_ref)
bd = simplehf.hf.orb.density(nb, c_ref)

af, bf = simplehf.hf.uhf.fock(h, g, ad, bd)
f = simplehf.hf.rohf.effective_fock(s, af, bf, ad, bd)

numpy.save('af', af)
numpy.save('bf', bf)
numpy.save('ad', ad)
numpy.save('bd', bd)

c = simplehf.hf.orb.coefficients(s, f)

print((numpy.abs(c) - numpy.abs(c_ref)).round(1))
print(numpy.linalg.norm(numpy.abs(c) - numpy.abs(c_ref)))

ad = simplehf.hf.orb.density(na, c)
bd = simplehf.hf.orb.density(nb, c)
af, bf = simplehf.hf.uhf.fock(h, g, ad, bd)

energy = simplehf.hf.uhf.energy(h, af, bf, ad, bd)

nuc_energy = simplehf.chem.nuc.energy(LABELS, COORDS)

print(repr(energy + nuc_energy))
