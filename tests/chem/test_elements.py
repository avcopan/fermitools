from fermitools.chem import elements


def test__charge():
    assert elements.charge('uuo') == 118
    assert elements.charge('Uuo') == 118
    assert elements.charge('UUO') == 118
