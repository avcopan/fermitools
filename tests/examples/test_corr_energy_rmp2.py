import fermitools.examples.corr_energy_rmp2 as example


def test__main():
    assert hasattr(example, 'main')
    example.main()
