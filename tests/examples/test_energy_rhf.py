import simplehf.examples.energy_rhf as example


def test__main():
    assert hasattr(example, 'main')
    example.main()
