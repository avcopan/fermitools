from simplehf.chem import elec


CHARGE = +1
SPIN = 1
LABELS = ("O", "H", "H")


def test__count():
    assert elec.count(labels=LABELS, charge=CHARGE) == 9


def test__count_alpha():
    assert elec.count_alpha(labels=LABELS, charge=CHARGE, spin=SPIN) == 5


def test__count_beta():
    assert elec.count_beta(labels=LABELS, charge=CHARGE, spin=SPIN) == 4
