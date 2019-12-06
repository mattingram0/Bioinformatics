import assignment
import numpy as np


def test_dynprog():
    quad_results1 = assignment.dynprog("ABCD",
                                       [[1, -5, -5, -5, -1],
                                       [-5, 1, -5, -5, -1],
                                       [-5, -5, 5, -5, -4],
                                       [-5, -5, -5, 6, -4],
                                       [-1, -1, -4, -4, -9]]
                                       ,
                                      "AAAAACCDDCCDDAAAAACC",
                                      "CCAAADDAAAACCAAADDCCAAAA")

    quad_results2 = assignment.dynprog("ABCD",
                                       [[1, -5, -5, -5, -1],
                                       [-5, 1, -5, -5, -1],
                                       [-5, -5, 5, -5, -4],
                                       [-5, -5, -5, 6, -4],
                                       [-1, -1, -4, -4, -9]]
                                       ,
                                      "AACAAADAAAACAADAADAAA",
                                      "CDCDDD")

    quad_results3 = assignment.dynprog("ABCD",
                                       [[1, -5, -5, -5, -1],
                                       [-5, 1, -5, -5, -1],
                                       [-5, -5, 5, -5, -4],
                                       [-5, -5, -5, 6, -4],
                                       [-1, -1, -4, -4, -9]]
                                       ,
                                      "DDCDDCCCDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCDDDC"
                                      "DADCDCDCDCD",
                                      "DDCDDCCCDCBCCCCDDDCDBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
                                      "BBBBBBBBBBBBBDCDCDCDCD")

    quad_results4 = assignment.dynprog("ABC",
                                       [[1, -1, -2, -1], [-1, 2, -4, -1],
                                       [-2, -4, 3, -2], [-1, -1, -2, 0]]
                                       ,
                                      "AABBAACA",
                                      "CBACCCBA")

    lin_results1 = assignment.dynproglin("ABCD",
                                         [[1, -5, -5, -5, -1],
                                      [-5, 1, -5, -5, -1],
                                      [-5, -5, 5, -5, -4],
                                      [-5, -5, -5, 6, -4],
                                      [-1, -1, -4, -4, -9]]
                                         ,
                                     "AAAAACCDDCCDDAAAAACC",
                                     "CCAAADDAAAACCAAADDCCAAAA")

    lin_results2 = assignment.dynproglin("ABCD",
                                         [[1, -5, -5, -5, -1],
                                      [-5, 1, -5, -5, -1],
                                      [-5, -5, 5, -5, -4],
                                      [-5, -5, -5, 6, -4],
                                      [-1, -1, -4, -4, -9]]
                                         ,
                                     "AACAAADAAAACAADAADAAA",
                                     "CDCDDD")

    lin_results3 = assignment.dynproglin("ABCD",
                                         [[1, -5, -5, -5, -1],
                                      [-5, 1, -5, -5, -1],
                                      [-5, -5, 5, -5, -4],
                                      [-5, -5, -5, 6, -4],
                                      [-1, -1, -4, -4, -9]]
                                         ,
                                     "DDCDDCCCDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCDDDC"
                                     "DADCDCDCDCD",
                                     "DDCDDCCCDCBCCCCDDDCDBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
                                     "BBBBBBBBBBBBBDCDCDCDCD")

    lin_results4 = assignment.dynproglin("ABC",
                                         [[1, -1, -2, -1], [-1, 2, -4, -1],
                                      [-2, -4, 3, -2], [-1, -1, -2, 0]]
                                         ,
                                     "AABBAACA",
                                     "CBACCCBA")

    assert quad_results1[0] == lin_results1[0] == 39.0
    assert all(x[0] == x[1] == x[2] for x in zip(quad_results1[1],
                                                 lin_results1[1],
                                                 [5, 6, 7, 8, 9, 10, 11, 12,
                                                  18, 19]))
    assert all(x[0] == x[1] == x[2] for x in zip(quad_results1[2],
                                                 lin_results1[2], [0, 1, 5,
                                                                   6, 11, 12,
                                                                   16, 17,
                                                                   18, 19]))
    assert quad_results2[0] == lin_results2[0] == 17.0
    assert all(x[0] == x[1] == x[2] for x in zip(quad_results2[1],
                                                 lin_results2[1], [2, 6, 11,
                                                                   14, 17]))
    assert all(x[0] == x[1] == x[2] for x in zip(quad_results2[2],
                                                 lin_results2[2], [0, 1, 2,
                                                                   3, 4]))
    assert quad_results3[0] == lin_results3[0] == 81.0
    assert all(x[0] == x[1] == x[2] for x in zip(quad_results3[1],
                                                 lin_results3[1], [0, 1, 2,
                                                                   3, 4, 5, 6,
                                                                   7, 8,
                                                                   9, 40, 41,
                                                                   42, 43, 44,
                                                                   45,
                                                                   46, 47, 48,
                                                                   50, 51, 52,
                                                                   53,
                                                                   54, 55, 56,
                                                                   57, 58]))
    assert all(x[0] == x[1] == x[2] for x in zip(quad_results3[2],
                                                 lin_results3[2], [0, 1, 2,
                                                                   3, 4, 5, 6,
                                                                   7,
                                                                   8, 9, 11,
                                                                   12, 13, 14,
                                                                   15,
                                                                   16, 17, 18,
                                                                   19, 61, 62,
                                                                   63,
                                                                   64, 65, 66,
                                                                   67, 68,
                                                                   69]))
    assert quad_results4[0] == lin_results4[0] == 5.0
    assert all(x[0] == x[1] == x[2] for x in zip(quad_results4[1],
                                                 lin_results4[1], [3, 5, 6]))
    assert all(x[0] == x[1] == x[2] for x in zip(quad_results4[2],
                                                 lin_results4[2], [1, 2, 3]))


if __name__ == "__main__":
    test_dynprog()
