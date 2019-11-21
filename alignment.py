import numpy as np
import sys
import ast


def dynprog(alphabet, scoring_matrix, sequence1, sequence2):
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)

    align_matrix = np.zeros(shape=(m + 1, n + 1))
    pointers = np.zeros(shape=(m + 1, n + 1))

    # First row (handle explicitly in the case the scoring matrix is
    # adversarial and assigns positive values for gaps)
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

    # First column (handle explicitly in the case the scoring matrix is
    # adversarial and assigns positive values for gaps)
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
                # ^ Start a new subsequence
                align_matrix[i + 1, j] + scoring_matrix[index['_'], index[k]],
                # ^ Align letter in seq2 to a gap (add gap to seq1)
                align_matrix[i, j] + scoring_matrix[index[l], index[k]],
                # ^ The letters in both seq (conservatively) match
                align_matrix[i, j + 1] + scoring_matrix[index[l], index['_']]
                # ^ Align letter in seq1 to a gap (add gap to seq2)
            ])

            pointers[i + 1, j + 1] = scores.argmax()  # Poor coding style -
            # relies on the order of the lines of code themselves
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

    # print("Final Matrix: ")
    # print(align_matrix, "\n")
    # print("Final Pointer Matrix")
    # print(pointers)

    # TODO - sloppy - keep track of the max value during the algo itself
    return [align_matrix.max(), seq1_indices, seq2_indices]


def dynproglin(alphabet, scoring_matrix, sequence1, sequence2):
    # TODO - make sure to iterate over the minimum of the two lengths -
    #  check this beforehand

    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)
    best_start = [0, 0]
    best_score = 0
    best_pos = [0, 0]

    # Iterate through the whole matrix, finding the maximum score and the
    # starting position of the path that gives that score, in linear space
    if m >= n:
        prev_start = [(0, i) for i in range(n + 1)]
        curr_start = [(0, i) for i in range(n + 1)]
        prev_row = np.zeros((1, n + 1))
        curr_row = np.zeros((1, n + 1))
        # TODO - make sure the first row/column are correct

        # Rest of matrix:
        for i, l in enumerate(sequence1):
            for j, k in enumerate(sequence2):
                left = curr_row[0, j] + scoring_matrix[index['_'], index[k]]
                diag = prev_row[0, j] + scoring_matrix[index[l], index[k]]
                up = prev_row[0, j + 1] + scoring_matrix[index[l], index['_']]

                score = max(0, left, diag, up)
                curr_row[0, j + 1] = score

                if score == 0:
                    curr_start[j + 1] = (i + 1, j + 1)
                elif score == left:
                    curr_start[j + 1] = curr_start[j]
                elif score == diag:
                    curr_start[j + 1] = prev_start[j]
                else:
                    curr_start[j + 1] = prev_start[j + 1]

                if score > best_score:
                    best_score = score
                    best_start = curr_start[j + 1]
                    best_pos = [i + 1, j + 1]

            # Reset at end of row
            prev_row = curr_row
            curr_row = np.zeros((1, n + 1))
            prev_start = curr_start
            prev_start[0] = (i + 1, 0)
            curr_start = [(0, 0) for i in range(n + 1)]

    else:
        prev_start = [(i, 0) for i in range(n + 1)]
        curr_start = [(i, 0) for i in range(n + 1)]
        prev_col = np.zeros((n + 1, 1))
        curr_col = np.zeros((n + 1, 1))

        # TODO - make sure the first row/column are correct

        # Rest of matrix:
        for i, l in enumerate(sequence2):
            for j, k in enumerate(sequence1):
                left = prev_col[j + 1, 0] + scoring_matrix[index['_'],index[l]]
                diag = prev_col[j, 0] + scoring_matrix[index[l], index[k]]
                up = curr_col[j, 0] + scoring_matrix[index[k], index['_']]

                score = max(0, left, diag, up)
                curr_col[j + 1, 0] = score

                if score == 0:
                    curr_start[j + 1] = (i + 1, j + 1)
                elif score == left:
                    curr_start[j + 1] = prev_start[j + 1]
                elif score == diag:
                    curr_start[j + 1] = prev_start[j]
                else:
                    curr_start[j + 1] = curr_start[j]

                if score > best_score:
                    best_score = score
                    best_start = curr_start[j + 1]
                    best_pos = [i + 1, j + 1]

            # Reset at end of col
            prev_col = curr_col
            curr_col = np.zeros((n + 1, 1))
            prev_start = curr_start
            prev_start[0] = (0, i + 1)
            curr_start = [(0, 0) for i in range(n + 1)]


def dynproglin_recursive(alphabet, scoring_matrix, start, end, seq1, seq2):

    # Base cases:
    if len(seq1) == 1:
        pass

    if len(seq2) == 1:
        pass

    return dynproglin_recursive(alphabet, scoring_matrix, new_start,
                                midpoint, seq1_head, seq2_head) +\
           [midpoint_index] +\
           dynproglin_recursive(alphabet, scoring_matrix, midpoint, new_end,
                                seq1_tail, seq2_tail)



def heuralign(alphabet, scoring_matrix, sequence1, sequence2):
    pass


def main():
    # Parse Input
    alphabet = sys.argv[1]
    scoring_matrix = ast.literal_eval(sys.argv[2])
    seq1 = sys.argv[3]
    seq2 = sys.argv[4]

    # Print format options for numpy
    np.set_printoptions(edgeitems=20, linewidth=100000)

    dynprog(alphabet, scoring_matrix, seq1, seq2)
    dynproglin(alphabet, scoring_matrix, seq1, seq2)
    # results = dynprog(alphabet, scoring_matrix, seq1, seq2)
    # print("Sequence 1: ", seq1)
    # print("Sequence 2: ", seq2)
    # print("Score: ", results[0])
    # print("Sequence 1 Indices: ", results[1])
    # print("Sequence 2 Indices: ", results[2])


if __name__ == "__main__":
    main()
