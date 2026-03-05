import numpy as np
import pytest

from weighted_levenshtein import dam_lev, lev, osa


@pytest.fixture
def weights():
    return {
        'iw': np.ones(128, dtype=np.float64),
        'dw': np.ones(128, dtype=np.float64),
        'sw': np.ones((128, 128), dtype=np.float64),
        'tw': np.ones((128, 128), dtype=np.float64),
    }


def _lev(w, x, y):
    return lev(x, y, w['iw'], w['dw'], w['sw'])


def _osa(w, x, y):
    return osa(x, y, w['iw'], w['dw'], w['sw'], w['tw'])


def _dl(w, x, y):
    return dam_lev(x, y, w['iw'], w['dw'], w['sw'], w['tw'])


def test_lev(weights):
    assert _lev(weights, '1234', '1234') == 0.0
    assert _lev(weights, '', '1234') == 4.0
    assert _lev(weights, '1234', '') == 4.0
    assert _lev(weights, '', '') == 0.0
    assert _lev(weights, '1234', '12') == 2.0
    assert _lev(weights, '1234', '14') == 2.0
    assert _lev(weights, '1111', '1') == 3.0


def test_lev_insert(weights):
    weights['iw'][ord('a')] = 5
    assert _lev(weights, '', 'a') == 5.0
    assert _lev(weights, 'a', '') == 1.0
    assert _lev(weights, '', 'aa') == 10.0
    assert _lev(weights, 'a', 'aa') == 5.0
    assert _lev(weights, 'aa', 'a') == 1.0
    assert _lev(weights, 'asdf', 'asdf') == 0.0
    assert _lev(weights, 'xyz', 'abc') == 3.0
    assert _lev(weights, 'xyz', 'axyz') == 5.0
    assert _lev(weights, 'x', 'ax') == 5.0


def test_lev_delete(weights):
    weights['dw'][ord('z')] = 7.5
    assert _lev(weights, '', 'z') == 1.0
    assert _lev(weights, 'z', '') == 7.5
    assert _lev(weights, 'xyz', 'zzxz') == 3.0
    assert _lev(weights, 'zzxzzz', 'xyz') == 18.0


def test_lev_substitute(weights):
    weights['sw'][ord('a'), ord('z')] = 1.2
    weights['sw'][ord('z'), ord('a')] = 0.1
    assert _lev(weights, 'a', 'z') == 1.2
    assert _lev(weights, 'z', 'a') == 0.1
    assert _lev(weights, 'a', '') == 1
    assert _lev(weights, '', 'a') == 1
    assert _lev(weights, 'asdf', 'zzzz') == 4.2
    assert _lev(weights, 'asdf', 'zz') == 4.0
    assert _lev(weights, 'asdf', 'zsdf') == 1.2
    assert _lev(weights, 'zsdf', 'asdf') == 0.1


def test_osa(weights):
    assert _osa(weights, '1234', '1234') == 0.0
    assert _osa(weights, '', '1234') == 4.0
    assert _osa(weights, '1234', '') == 4.0
    assert _osa(weights, '', '') == 0.0
    assert _osa(weights, '1234', '12') == 2.0
    assert _osa(weights, '1234', '14') == 2.0
    assert _osa(weights, '1111', '1') == 3.0


def test_osa_insert(weights):
    weights['iw'][ord('a')] = 5
    assert _osa(weights, '', 'a') == 5.0
    assert _osa(weights, 'a', '') == 1.0
    assert _osa(weights, '', 'aa') == 10.0
    assert _osa(weights, 'a', 'aa') == 5.0
    assert _osa(weights, 'aa', 'a') == 1.0
    assert _osa(weights, 'asdf', 'asdf') == 0.0
    assert _osa(weights, 'xyz', 'abc') == 3.0
    assert _osa(weights, 'xyz', 'axyz') == 5.0
    assert _osa(weights, 'x', 'ax') == 5.0


def test_osa_delete(weights):
    weights['dw'][ord('z')] = 7.5
    assert _osa(weights, '', 'z') == 1.0
    assert _osa(weights, 'z', '') == 7.5
    assert _osa(weights, 'xyz', 'zzxz') == 3.0
    assert _osa(weights, 'zzxzzz', 'xyz') == 18.0


def test_osa_substitute(weights):
    weights['sw'][ord('a'), ord('z')] = 1.2
    weights['sw'][ord('z'), ord('a')] = 0.1
    assert _osa(weights, 'a', 'z') == 1.2
    assert _osa(weights, 'z', 'a') == 0.1
    assert _osa(weights, 'a', '') == 1
    assert _osa(weights, '', 'a') == 1
    assert _osa(weights, 'asdf', 'zzzz') == 4.2
    assert _osa(weights, 'asdf', 'zz') == 4.0
    assert _osa(weights, 'asdf', 'zsdf') == 1.2
    assert _osa(weights, 'zsdf', 'asdf') == 0.1


def test_osa_transpose(weights):
    weights['tw'][ord('a'), ord('z')] = 1.5
    weights['tw'][ord('z'), ord('a')] = 0.5
    assert _osa(weights, 'az', 'za') == 1.5
    assert _osa(weights, 'za', 'az') == 0.5
    assert _osa(weights, 'az', 'zfa') == 3
    assert _osa(weights, 'azza', 'zaaz') == 2
    assert _osa(weights, 'zaaz', 'azza') == 2
    assert _osa(weights, 'azbza', 'zabaz') == 2
    assert _osa(weights, 'zabaz', 'azbza') == 2
    assert _osa(weights, 'azxza', 'zayaz') == 3
    assert _osa(weights, 'zaxaz', 'azyza') == 3


def test_dl(weights):
    assert _dl(weights, '', '') == 0
    assert _dl(weights, '', 'a') == 1
    assert _dl(weights, 'a', '') == 1
    assert _dl(weights, 'a', 'b') == 1
    assert _dl(weights, 'a', 'ab') == 1
    assert _dl(weights, 'ab', 'ba') == 1
    assert _dl(weights, 'ab', 'bca') == 2
    assert _dl(weights, 'bca', 'ab') == 2
    assert _dl(weights, 'ab', 'bdca') == 3
    assert _dl(weights, 'bdca', 'ab') == 3


def test_dl_transpose(weights):
    weights['iw'][ord('c')] = 1.9
    assert _dl(weights, 'ab', 'bca') == 2.9
    assert _dl(weights, 'ab', 'bdca') == 3.9
    assert _dl(weights, 'bca', 'ab') == 2


def test_dl_transpose2(weights):
    weights['dw'][ord('c')] = 1.9
    assert _dl(weights, 'bca', 'ab') == 2.9
    assert _dl(weights, 'bdca', 'ab') == 3.9
    assert _dl(weights, 'ab', 'bca') == 2


def test_dl_transpose3(weights):
    weights['tw'][ord('a'), ord('b')] = 1.5
    assert _dl(weights, 'ab', 'bca') == 2.5
    assert _dl(weights, 'bca', 'ab') == 2


def test_dl_transpose4(weights):
    weights['tw'][ord('b'), ord('a')] = 1.5
    assert _dl(weights, 'ab', 'bca') == 2
    assert _dl(weights, 'bca', 'ab') == 2.5


def test_lev_defaults():
    assert lev('1234', '1234') == 0.0
    assert lev('', '1234') == 4.0
    assert lev('1234', '') == 4.0
    assert lev('', '') == 0.0
    assert lev('1234', '12') == 2.0
    assert lev('1234', '14') == 2.0
    assert lev('1111', '1') == 3.0


def test_osa_defaults():
    assert osa('1234', '1234') == 0.0
    assert osa('', '1234') == 4.0
    assert osa('1234', '') == 4.0
    assert osa('', '') == 0.0
    assert osa('1234', '12') == 2.0
    assert osa('1234', '14') == 2.0
    assert osa('1111', '1') == 3.0


def test_dl_defaults():
    assert dam_lev('', '') == 0
    assert dam_lev('', 'a') == 1
    assert dam_lev('a', '') == 1
    assert dam_lev('a', 'b') == 1
    assert dam_lev('a', 'ab') == 1
    assert dam_lev('ab', 'ba') == 1
    assert dam_lev('ab', 'bca') == 2
    assert dam_lev('bca', 'ab') == 2
    assert dam_lev('ab', 'bdca') == 3
    assert dam_lev('bdca', 'ab') == 3
