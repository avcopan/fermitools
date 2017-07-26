import fermitools.math.spinorb.mo as mo


def test__sort_order():
    assert mo.sort_order(dim=4, na=2, nb=2) == (0, 1, 4, 5, 2, 3, 6, 7)
    assert mo.sort_order(dim=5, na=3, nb=2) == (0, 1, 2, 5, 6, 3, 4, 7, 8, 9)
