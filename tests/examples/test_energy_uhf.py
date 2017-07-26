import fermitools.examples.energy_uhf as example


def test__main():
    assert hasattr(example, 'main')
    example.main()
