from mathfuncs import inc, dec, is_positive


def test_inc_basic():
    assert inc(1) == 2
    assert inc(5) == 6


def test_dec_basic():
    assert dec(2) == 1
    assert dec(0) == -1


def test_is_positive():
    assert is_positive(5)
    assert not is_positive(-3)
    assert not is_positive(0)
