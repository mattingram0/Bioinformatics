import alignment
import numpy as np


def test_dynprog():
    assert alignment.dynprog("ABCD",
                             [[1, -5, -5, -5, -1], [-5, 1, -5, -5, -1],
                              [-5, -5, 5, -5, -4], [-5, -5, -5, 6, -4],
                              [-1, -1, -4, -4, -9]]
                             ,
                             "AAAAACCDDCCDDAAAAACC",
                             "CCAAADDAAAACCAAADDCCAAAA")[0] == 39.0

    assert alignment.dynprog("ABCD",
                             [[1, -5, -5, -5, -1], [-5, 1, -5, -5, -1],
                              [-5, -5, 5, -5, -4], [-5, -5, -5, 6, -4],
                              [-1, -1, -4, -4, -9]]
                             ,
                             "AACAAADAAAACAADAADAAA",
                             "CDCDDD")[0] == 17.0

    assert alignment.dynprog("ABCD",
                             [[1, -5, -5, -5, -1], [-5, 1, -5, -5, -1],
                              [-5, -5, 5, -5, -4], [-5, -5, -5, 6, -4],
                              [-1, -1, -4, -4, -9]]
                             ,
                             "DDCDDCCCDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCDDDC"
                             "DADCDCDCDCD",
                             "DDCDDCCCDCBCCCCDDDCDBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
                             "BBBBBBBBBBBBBDCDCDCDCD")[0] == 81.0

    assert alignment.dynprog("ABC",
                             [[1, -1, -2, -1], [-1, 2, -4, -1],
                              [-2, -4, 3, -2], [-1, -1, -2, 0]]
                             ,
                             "AABBAACA",
                             "CBACCCBA")[0] == 5.0


if __name__ == "__main__":
    test_dynprog()
