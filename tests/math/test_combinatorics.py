import fermitools.math.combinatorics as comb


def test__signature():
    assert comb.signature(p='abcd', i='abcd') == +1
    assert comb.signature(p='abdc', i='abcd') == -1
    assert comb.signature(p='acbd', i='abcd') == -1
    assert comb.signature(p='acdb', i='abcd') == +1
    assert comb.signature(p='adbc', i='abcd') == +1
    assert comb.signature(p='adcb', i='abcd') == -1
    assert comb.signature(p='bacd', i='abcd') == -1
    assert comb.signature(p='badc', i='abcd') == +1
    assert comb.signature(p='bcad', i='abcd') == +1
    assert comb.signature(p='bcda', i='abcd') == -1
    assert comb.signature(p='bdac', i='abcd') == -1
    assert comb.signature(p='bdca', i='abcd') == +1
    assert comb.signature(p='cabd', i='abcd') == +1
    assert comb.signature(p='cadb', i='abcd') == -1
    assert comb.signature(p='cbad', i='abcd') == -1
    assert comb.signature(p='cbda', i='abcd') == +1
    assert comb.signature(p='cdab', i='abcd') == +1
    assert comb.signature(p='cdba', i='abcd') == -1
    assert comb.signature(p='dabc', i='abcd') == -1
    assert comb.signature(p='dacb', i='abcd') == +1
    assert comb.signature(p='dbac', i='abcd') == +1
    assert comb.signature(p='dbca', i='abcd') == -1
    assert comb.signature(p='dcab', i='abcd') == -1
    assert comb.signature(p='dcba', i='abcd') == +1


def test__permuter():
    sentence = 'The quick brown fox jumps over the lazy dog.'
    scramble = comb.permuter(p='bcdefa', i='abcdef')
    unscramble = comb.permuter(p='fabcde', i='abcdef')

    assert ''.join(scramble(sentence)) == ('Thf quidk crown aox jumps ovfr '
                                           'thf lbzy eog.')

    assert ''.join(unscramble(scramble(sentence))) == sentence


def test__riffle_shuffles():

    assert set(comb.riffle_shuffles((0, 1, 2, 3), ksizes=(4,))) == {
                (0, 1, 2, 3)
            }

    assert set(comb.riffle_shuffles((0, 1, 2, 3), ksizes=(3, 1))) == {
                (0, 1, 2, 3), (0, 1, 3, 2), (0, 3, 1, 2), (3, 0, 1, 2)
            }

    assert set(comb.riffle_shuffles((0, 1, 2, 3), ksizes=(2, 2))) == {
                (0, 1, 2, 3), (0, 2, 1, 3), (2, 0, 1, 3), (0, 2, 3, 1),
                (2, 0, 3, 1), (2, 3, 0, 1)
            }

    assert set(comb.riffle_shuffles((0, 1, 2, 3), ksizes=(2, 1, 1))) == {
                (0, 1, 2, 3), (0, 2, 1, 3), (2, 0, 1, 3), (0, 1, 3, 2),
                (0, 3, 1, 2), (3, 0, 1, 2), (0, 2, 3, 1), (2, 0, 3, 1),
                (0, 3, 2, 1), (3, 0, 2, 1), (2, 3, 0, 1), (3, 2, 0, 1)
            }

    assert set(comb.riffle_shuffles((0, 1, 2, 3), ksizes=(1, 1, 1, 1))) == {
                (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1),
                (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2),
                (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
                (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0),
                (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1),
                (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0)
            }
