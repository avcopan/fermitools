import fermitools.math.tensoralg.srt as srt

import pytest
import numpy


def test__einsum_argsort():
    a1 = numpy.random.random((1, 2))
    a2 = numpy.random.random((1, 2, 10))
    b = numpy.random.random((2, 3))
    c = numpy.random.random((3, 4))
    assert srt.einsum_argsort('ij...,jk,kl', a1, b, c) == (0, 1, 2)
    assert srt.einsum_argsort('ij...,jk,kl', a2, b, c) == (1, 2, 0)


def test__cost_function():
    cost1 = srt.cost_function(({'i': 1, 'k': 2, '#': 1},
                               {'k': 2, 'l': 3, '#': 1},
                               {'l': 3, 'j': 4, '#': 1}))
    cost2 = srt.cost_function(({'i': 1, 'k': 2, '#': 10},
                               {'k': 2, 'l': 3, '#': 1},
                               {'l': 3, 'j': 4, '#': 1}))
    cost3 = srt.cost_function(({'i': 1, 'k': 2, '#': 1},
                               {'k': 2, 'l': 3, '#': 10},
                               {'l': 3, 'j': 4, '#': 1}))
    assert cost1((0, 1, 2)) == 18
    assert cost1((1, 2, 0)) == 32
    assert cost2((0, 1, 2)) == 180
    assert cost2((1, 2, 0)) == 104
    assert cost3((0, 1, 2)) == 180
    assert cost3((1, 2, 0)) == 320


def test__dimdict():
    shp = (1, 2, 3, 4, 5)
    assert srt.dimdict(shp, 'ijklm') == {
            'i': 1, 'j': 2, 'k': 3, 'l': 4, 'm': 5, '#': 1}
    assert srt.dimdict(shp, '#ijklm') == {
            'i': 1, 'j': 2, 'k': 3, 'l': 4, 'm': 5, '#': 1}
    assert srt.dimdict(shp, 'ij#klm') == {
            'i': 1, 'j': 2, 'k': 3, 'l': 4, 'm': 5, '#': 1}
    assert srt.dimdict(shp, 'ijklm#') == {
            'i': 1, 'j': 2, 'k': 3, 'l': 4, 'm': 5, '#': 1}
    assert srt.dimdict(shp, '#ijkl') == {
            '#': 1, 'i': 2, 'j': 3, 'k': 4, 'l': 5}
    assert srt.dimdict(shp, 'ij#kl') == {
            'i': 1, 'j': 2, '#': 3, 'k': 4, 'l': 5}
    assert srt.dimdict(shp, 'ijkl#') == {
            'i': 1, 'j': 2, 'k': 3, 'l': 4, '#': 5}
    assert srt.dimdict(shp, '#ijk') == {
            '#': 2, 'i': 3, 'j': 4, 'k': 5}
    assert srt.dimdict(shp, 'ij#k') == {
            'i': 1, 'j': 2, '#': 12, 'k': 5}
    assert srt.dimdict(shp, 'ijk#') == {
            'i': 1, 'j': 2, 'k': 3, '#': 20}
    assert srt.dimdict(shp, '#i') == {
            '#': 24, 'i': 5}
    assert srt.dimdict(shp, 'i#') == {
            'i': 1, '#': 120}
    assert srt.dimdict(shp, '#') == {
            '#': 120}

    with pytest.raises(AssertionError):
        srt.dimdict(shp, 'ijkl')

    with pytest.raises(AssertionError):
        srt.dimdict(shp, 'ijklmn#')

    with pytest.raises(AssertionError):
        srt.dimdict(shp, 'ijk##')


if __name__ == '__main__':
    test__einsum_argsort()
