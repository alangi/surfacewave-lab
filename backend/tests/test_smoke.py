def test_imports():
    import swmam  # noqa: F401
    import swinversion  # noqa: F401

    assert swmam is not None
    assert swinversion is not None
