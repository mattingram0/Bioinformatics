import numpy as np
from math import floor
import sys
import ast


# TODO - add var_name = none for numpy arrays and other variables to tell
#  Python GC that the memory can be reused

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
    # TODO - add the swap back of the sequences

    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)
    best_start = [0, 0]
    best_score = 0
    best_end = [0, 0]
    swapped = False

    # Swap the sequences so that the longest sequence is always sequence 1,
    # which is on the left of the matrix. i.e number of rows > no. of cols
    if m < n:
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        temp = m
        m = n
        n = temp
        swapped = True

    # Iterate through the whole matrix, finding the maximum score and the
    # starting position of the path that gives that score, in linear space
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
                best_end = [i + 1, j + 1]

        # Reset at end of row
        prev_row = curr_row
        curr_row = np.zeros((1, n + 1))
        prev_start = curr_start
        prev_start[0] = (i + 1, 0)
        curr_start = [(0, 0) for i in range(n + 1)]

    # Swap back if we swapped the sequences
    if swapped:
        best_start.reverse()
        best_end.reverse()
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp

    # Start the recursion - TODO finish - need to trim the sequences passed
    dynproglin_recursive(alphabet, scoring_matrix,
                         sequence1[best_start[0] - 1: best_end[0]],
                         sequence2[best_start[1] - 1: best_end[1]],
                         best_start)


def dynproglin_recursive(alphabet, scoring_matrix, sequence1, sequence2,
                         start):
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)
    swapped = False

    # Swap if sequence2 is longer than sequence1 - TODO remember to switch back
    if m < n:
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        temp = m
        m = n
        n = temp
        swapped = True

    mid_col = floor(n / 2)

    # Base case:
    if len(sequence1) == 0:
        return []

    # TODO - check if one off error in case of errors
    prev_mid = [0 for i in range(m - mid_col + 1)]
    curr_mid = [0 for i in range(m - mid_col + 1)]
    prev_row = np.zeros((1, n + 1))
    curr_row = np.zeros((1, n + 1))

    # Rest of matrix:
    for i, l in enumerate(sequence1):
        for j, k in enumerate(sequence2):
            left = curr_row[0, j] + scoring_matrix[index['_'], index[k]]
            diag = prev_row[0, j] + scoring_matrix[index[l], index[k]]
            up = prev_row[0, j + 1] + scoring_matrix[index[l], index['_']]

            score = max(left, diag, up)  # Trying to find a global
            # alignment now, so no 0 included
            curr_row[0, j + 1] = score

            # We only need to keep track of the mid_value for the half of
            # the matrix after the mid_col
            if j >= mid_col:
                # If we are currently on the middle column
                if j == mid_col:
                    # If it came from the left, then curr_mid[j + 1] = i + 1
                    # If it came from the diag, then curr_mid[j + 1] = i
                    # If it came from above, then curr_mid[j + 1] = prev_mid[i + 1]
                    if score == left:
                        curr_mid[1] = i + 1
                    elif score == diag:
                        curr_mid[1] = i
                    else:
                        curr_mid[1] = prev_mid[i + 1]
                # If we're not on the middle column, then propagate the
                # middle along the path
                else:
                    if score == left:
                        curr_mid[j - mid_col + 1] = curr_mid[j - mid_col]
                    elif score == diag:
                        curr_mid[j - mid_col + 1] = prev_mid[j - mid_col]
                    else:
                        curr_mid[j - mid_col + 1] = prev_mid[j - mid_col + 1]

            # TODO - check this
            # Reset at end of row
            prev_row = curr_row
            curr_row = np.zeros((1, n + 1))
            prev_mid = curr_mid
            curr_mid = [0 for i in range(len(sequence2) - mid_col + 1)]

    # TODO - check this
    mid_point = start + [curr_mid[m - mid_col], mid_col]

    # Swap back
    if swapped:
        mid_point.reverse()
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp

    return dynproglin_recursive(
        alphabet, scoring_matrix, sequence1[:mid_point[0]],
        sequence2[:mid_point[1]], start
    ) + [mid_point] + dynproglin_recursive(
        alphabet, scoring_matrix, sequence1[mid_point[0]:],
        sequence2[mid_point[1]:], mid_point
    )


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
