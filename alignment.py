import numpy as np
from math import floor
from itertools import product
import sys
import ast


# TODO - add var_name = none for numpy arrays and other variables to tell
#  Python GC that the memory can be reused

def dynprog(alphabet, scoring_matrix, sequence1, sequence2):
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    best_score = 0
    m = len(sequence1)
    n = len(sequence2)

    align_matrix = np.zeros(shape=(m + 1, n + 1))
    pointers = np.zeros(shape=(m + 1, n + 1))

    # First row
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

    # First column
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

    # Rest of the matrix:
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

            score = scores.max()
            pointers[i + 1, j + 1] = scores.argmax()  # Poor style

            if score > best_score:
                best_score = score

            align_matrix[i + 1, j + 1] = score

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

    # TODO - sloppy - keep track of the max value during the algo itself
    return [align_matrix.max(), seq1_indices, seq2_indices]


def dynproglin(alphabet, scoring_matrix, sequence1, sequence2):
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)
    best_start = [0, 0]
    best_score = 0
    best_end = [0, 0]
    swapped = False

    # Ensure we use the smaller of the two to iterate through, reducing memory
    if m < n:
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        temp = m
        m = n
        n = temp
        swapped = True

    # Used to keep track of the score and start point throughout the matrix
    prev_start = [[0, i] for i in range(n + 1)]
    curr_start = [[0, i] for i in range(n + 1)]
    prev_row = np.zeros((1, n + 1))
    curr_row = np.zeros((1, n + 1))

    # Take care of the initial row
    for j, k in enumerate(sequence2):
        prev_row[0, j + 1] = max(0, prev_row[0, j] + scoring_matrix[index[k]][
            index['_']])

    # Iterate through the rest of matrix:
    for i, l in enumerate(sequence1):
        for j, k in enumerate(sequence2):
            left = curr_row[0, j] + scoring_matrix[index['_'], index[k]]
            diag = prev_row[0, j] + scoring_matrix[index[l], index[k]]
            up = prev_row[0, j + 1] + scoring_matrix[index[l], index['_']]

            score = max(0, left, diag, up)
            curr_row[0, j + 1] = score

            if score == left:
                if curr_row[0, j] == 0:
                    curr_start[j + 1] = [i + 1, j + 1]
                else:
                    curr_start[j + 1] = list(curr_start[j])
            elif score == diag:
                if prev_row[0, j] == 0:
                    curr_start[j + 1] = [i + 1, j + 1]
                else:
                    curr_start[j + 1] = list(prev_start[j])
            else:
                if prev_row[0, j + 1] == 0:
                    curr_start[j + 1] = [i + 1, j + 1]
                else:
                    curr_start[j + 1] = list(prev_start[j + 1])

            if score > best_score:
                best_score = score
                best_start = list(curr_start[j + 1])
                best_end = [i + 1, j + 1]

        # Reset at end of row
        prev_row = curr_row
        curr_row = np.zeros((1, n + 1))

        prev_start = curr_start
        prev_start[0] = [i + 1, 0]
        curr_start = [[0, 0] for i in range(n + 1)]

    # Swap back if we swapped the sequences
    if swapped:
        best_start.reverse()
        best_end.reverse()
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp

    # Recursively break the matrix in half, finding the midpoint through
    # which the optimal path passes through in that submatrix
    coords = [best_start] + dynproglin_recursive(alphabet, scoring_matrix,
                                                 sequence1[
                                                 max(best_start[0] - 1, 0):
                                                 best_end[0]],
                                                 sequence2[
                                                 max(best_start[1] - 1, 0):
                                                 best_end[1]],
                                                 best_start) + [best_end]

    # Rebuild the alignment indices from the coordinates
    seq1_indices = [coords[0][0] - 1]
    seq2_indices = [coords[0][1] - 1]

    for i in range(len(coords) - 1):
        if coords[i + 1][0] - coords[i][0] == 1 and coords[i + 1][1] - \
                coords[i][1] == 1:
            seq1_indices.append(coords[i + 1][0] - 1)
            seq2_indices.append(coords[i + 1][1] - 1)

    return [best_score, seq1_indices, seq2_indices]


def dynproglin_recursive(alphabet, scoring_matrix, sequence1, sequence2,
                         start):
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)
    best_score = 0
    swapped = False

    # Ensure we use the smaller of the two to iterate through, reducing memory
    if m < n:
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        temp = m
        m = n
        n = temp
        swapped = True

    mid_col = int(floor(n / 2)) + 1

    # Base cases:
    if len(sequence2) == 2 or len(sequence1) == 2:
        return []

    if len(sequence1) == 1 and len(sequence2) == 1:
        return []

    # Used to keep track of on which row the middle column is joined and left
    prev_mid = [[0, 0] for i in range(n - mid_col + 1)]
    curr_mid = [[0, 0] for i in range(n - mid_col + 1)]

    prev_row = np.zeros((1, n + 1))
    curr_row = np.zeros((1, n + 1))

    # Used to keep track of on which row we leave the start & join the end cols
    prev_start = [0 for i in range(n + 1)]
    curr_start = [0 for i in range(n + 1)]
    curr_end = 0

    # TODO - ask LOUIS WHY WE DON'T NEED A ZERO HERE
    # prev_row[0, 0] = scoring_matrix[index['_'], index['_']]
    for j, k in enumerate(sequence2):
        prev_row[0, j + 1] = prev_row[0, j] + scoring_matrix[index[k]][
            index['_']]

    # Rest of matrix:
    for i, l in enumerate(sequence1):
        # TODO - ADD THIS LINE TO THE LOCAL INITIAL RUN - CANNOT ASSUME THAT
        #  THE SCORING MATRIX WILL BE NICE
        curr_row[0, 0] = prev_row[0, 0] + scoring_matrix[index[l]][index['_']]
        curr_start[1] = i + 1
        for j, k in enumerate(sequence2):
            left = curr_row[0, j] + scoring_matrix[index['_'], index[k]]
            diag = prev_row[0, j] + scoring_matrix[index[l], index[k]]
            up = prev_row[0, j + 1] + scoring_matrix[index[l], index['_']]

            score = max(left, diag, up)
            curr_row[0, j + 1] = score

            # Keep track of when we leave the start column
            if j >= 1:
                if score == left:
                    curr_start[j + 1] = curr_start[j]
                elif score == diag:
                    curr_start[j + 1] = prev_start[j]
                else:
                    curr_start[j + 1] = prev_start[j + 1]

            # Keep track of when we join and leave the middle column
            if j >= mid_col - 1:
                # Joining the middle column
                if j == mid_col - 1:
                    if score == left or score == diag:
                        curr_mid[0][0] = i + 1
                    else:
                        curr_mid[0][0] = prev_mid[0][0]

                # Leaving the middle column
                elif j == mid_col:
                    if score == left:
                        curr_mid[1] = [curr_mid[0][0], i + 1]
                    elif score == diag:
                        curr_mid[1] = [prev_mid[0][0], i]
                    else:
                        curr_mid[1] = list(prev_mid[1])

                # Propagating the middle column coordinates
                else:
                    if score == left:
                        curr_mid[j - mid_col + 1] = list(curr_mid[j - mid_col])
                    elif score == diag:
                        curr_mid[j - mid_col + 1] = list(prev_mid[j - mid_col])
                    else:
                        curr_mid[j - mid_col + 1] = list(prev_mid[j - mid_col
                                                                  + 1])

            # Keep track of when we join the final column
            if j == n - 1:
                if score == left or score == diag:
                    curr_end = i + 1

            # Keep track of best score
            if score > best_score:
                best_score = score

        # Reset at end of row
        prev_row = curr_row
        curr_row = np.zeros((1, n + 1))
        prev_mid = curr_mid
        curr_mid = [[0, 0] for i in range(n - mid_col + 1)]
        prev_start = curr_start
        curr_start = [0 for p in range(n + 1)]

    # Create the coordinates of the cells in the start, middle and end columns
    mid_points = []
    start_points = []
    end_points = []

    for i in range(1, prev_start[n] + 1):
        start_points.append([i, 1])

    for i in range(curr_end, m + 1):
        end_points.append([i, n])

    for i in range(prev_mid[n - mid_col][0], prev_mid[n - mid_col][1] + 1):
        mid_points.append([i, mid_col])

    # If we swapped the sequences swap them back, and transpose all coordinates
    if swapped:
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp

        for point in start_points + mid_points + end_points:
            point.reverse()

    # Split the two sequences into subsequences for the next recursive call
    seq1_head = sequence1[start_points[-1][0] - 1:mid_points[0][0]]
    seq2_head = sequence2[start_points[-1][1] - 1:mid_points[0][1]]
    seq1_tail = sequence1[mid_points[-1][0] - 1:end_points[0][0]]
    seq2_tail = sequence2[mid_points[-1][1] - 1:end_points[0][1]]

    # Convert the local coordinates to coordinates of the whole matrix
    real_mid_points = []
    real_start_points = []
    real_end_points = []

    for point in start_points:
        real_start_points.append([sum(x) - 1 for x in zip(start, point)])

    for point in end_points:
        real_end_points.append([sum(x) - 1 for x in zip(start, point)])

    for point in mid_points:
        real_mid_points.append([sum(x) - 1 for x in zip(start, point)])

    # Find the midpoint of the smaller submatrices
    return real_start_points[1:] + dynproglin_recursive(
        alphabet, scoring_matrix, seq1_head,
        seq2_head, real_start_points[-1]
    ) + real_mid_points + dynproglin_recursive(
        alphabet, scoring_matrix, seq1_tail,
        seq2_tail, real_mid_points[-1]
    ) + real_end_points[:-1]


def heuralign(alphabet, scoring_matrix, sequence1, sequence2):
    swapped = False
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    k_word_list = []

    # BLAST Parameters
    k = 3  # Length of the k-words
    T = 5  # No. of high-scoring tuples to keep
    # TODO assume scores are normally distributed and choose only the top 5%

    # Swap sequences
    if len(sequence1) < len(sequence2):
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        swapped = True

    m = len(sequence1)
    n = len(sequence2)

    # Generate the k-word list
    for i in range(n - k):
        k_word_list.append(sequence2[i:i + k])

    k_word_list = list(set(k_word_list))

    # Generate the word neighbours
    k_word_neighbours = {k: [] for k in k_word_list}
    k_tuples = [''.join(i) for i in product(alphabet, repeat=3)]

    for word in k_word_list:
        for tup in k_tuples:
            score = 0

            for i in range(k):
                score += scoring_matrix[index[word[i]], index[tup[i]]]

            k_word_neighbours[word].append((tup, score))

    # Keep only the best T k_word_neighbours of each k_tuple
    for k, v in k_word_neighbours.items():
        k_word_neighbours[k] = sorted(
            v, key=lambda tup_score: tup_score[1]
        )[-T:]

    for i in range(n - k):
        for j in range(m - k):
            pass  # TODO YOU ARE HERE
            # if sequence1[j: j + k] in k_word_neighbours[sequence2[i: i + k]]
                # Make note of the (i, j) location and the score
                # Also now make a dictionary of (i - j) : count, and choose
                # the top however many diagonals to greedily expand.

    # When expanding, expand in both directions. If the
    # accumulated score ever goes negative, for one direction, stop
    # Make note of the end (i', j') for that sequence.
    # Each one needs to be a [(i, j), (i', j'), score] pair - does it?

    return 0


def main():
    # Parse Input
    alphabet = sys.argv[1]
    scoring_matrix = ast.literal_eval(sys.argv[2])
    seq1 = sys.argv[3]
    seq2 = sys.argv[4]

    # Print format options for numpy
    np.set_printoptions(edgeitems=20, linewidth=100000)

    # Run the heuristic alignment
    heur_results = heuralign(alphabet, scoring_matrix, seq1, seq2)

    # # Run the quadratic and linear space algorithms
    # quad_results = dynprog(alphabet, scoring_matrix, seq1, seq2)
    # lin_results = dynproglin(alphabet, scoring_matrix, seq1, seq2)
    #
    # # Print results
    # print("Sequence 1: ", seq1)
    # print("Sequence 2: ", seq2)
    # print("Score (Quadratic): ", quad_results[0])
    # print("Score (Linear):    ", lin_results[0])
    # print("Sequence 1 Indices (Quadratic):", quad_results[1])
    # print("Sequence 1 Indices (Linear):   ", lin_results[1])
    # print("Sequence 2 Indices (Quadratic):", quad_results[2])
    # print("Sequence 2 Indices (Linear):   ", lin_results[2])


if __name__ == "__main__":
    main()
