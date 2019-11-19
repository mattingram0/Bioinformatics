import numpy as np
import sys
import ast


def dynprog(alphabet, scoring_matrix, sequence1, sequence2):
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)

    align_matrix = np.zeros(shape=(m + 1, n + 1))
    pointers = np.zeros(shape=(m + 1, n + 1))

    # First row:
    for i, l in enumerate(sequence1):
        score = max(
            0,
            align_matrix[i, 0] + scoring_matrix[index[l]][index['_']]
        )

        if score == 0:
            pointers[i + 1, 0] = 0
        else:
            pointers[i + 1, 0] = 1

        align_matrix[i + 1, 0] = score

    # First column:
    for j, k in enumerate(sequence2):
        score = max(
            0,
            align_matrix[0, j] + scoring_matrix[index[k]][index['_']]
        )

        if score == 0:
            pointers[0, j + 1] = 0
        else:
            pointers[0, j + 1] = 3

        align_matrix[0, j + 1] = score

    # Rest of matrix:
    for i, l in enumerate(sequence1):
        for j, k in enumerate(sequence2):
            scores = np.array([
                0,
                align_matrix[i + 1, j] + scoring_matrix[index['_'], index[k]],
                align_matrix[i, j] + scoring_matrix[index[l], index[k]],
                align_matrix[i, j + 1] + scoring_matrix[index[l], index['_']]
            ])

            pointers[i + 1, j + 1] = scores.argmax()
            align_matrix[i + 1, j + 1] = scores.max()

    # Form the indices
    i, j = np.unravel_index(align_matrix.argmax(), align_matrix.shape)
    direction = pointers[i, j]
    seq1_indices = []
    seq2_indices = []

    # Need to handle the case where the maximum is 0 and there isn't a
    # previous direction?

    while direction != 0:
        if direction == 1:
            j = j - 1
        elif direction == 2:
            seq1_indices.append(i - 1)
            seq2_indices.append(j - 1)
            i = i - 1
            j = j - 1
        else:
            i = i - 1

        direction = pointers[i, j]

    seq1_indices.reverse()
    seq2_indices.reverse()

    print("Sequence 1: ", sequence1)
    print("Sequence 2: ", sequence2)
    print("Score: ", align_matrix.max())
    print("Sequence 1 Indices: ", seq1_indices)
    print("Sequence 2 Indices: ", seq2_indices)
    print("Score Matrix: ", align_matrix)
    print("Pointer Matrix: ", pointers)


def dynproglin(alphabet, scoring_matrix, sequence1, sequence2):
    pass


def heuralign(alphabet, scoring_matrix, sequence1, sequence2):
    pass


def main():
    # Parse Input
    alphabet = sys.argv[1]
    scoring_matrix = np.array(ast.literal_eval(sys.argv[2]))
    seq1 = sys.argv[3]
    seq2 = sys.argv[4]

    dynprog(alphabet, scoring_matrix, seq1, seq2)


main()
