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

    print("Final Matrix: ")
    print(align_matrix, "\n")
    print("Final Pointer Matrix")
    print(pointers)

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
    prev_start = [[0, i] for i in range(n + 1)]
    curr_start = [[0, i] for i in range(n + 1)]
    prev_row = np.zeros((1, n + 1))
    curr_row = np.zeros((1, n + 1))

    # TODO - make sure the first row/column are correct - in the local
    #  alignment case, the first row is always 0 only if the scoring matrix
    #  is normal - if he feeds in a scoring matrix which gives positive
    #  values for a skipped letter, then we may run into issues without the
    #  following?
    prev_row[0, 0] = scoring_matrix[index['_'], index['_']]
    for j, k in enumerate(sequence2):
        prev_row[0, j + 1] = max(0, prev_row[0, j] + scoring_matrix[index[k]][
            index['_']])

    # Rest of matrix:
    for i, l in enumerate(sequence1):
        for j, k in enumerate(sequence2):
            left = curr_row[0, j] + scoring_matrix[index['_'], index[k]]
            diag = prev_row[0, j] + scoring_matrix[index[l], index[k]]
            up = prev_row[0, j + 1] + scoring_matrix[index[l], index['_']]

            score = max(0, left, diag, up)
            curr_row[0, j + 1] = score

            # TODO - start position is always one before it should be
            # TODO - think solved this,
            if score == left:
                if curr_row[0, j] == 0:
                    curr_start[j + 1] = [i + 1, j + 1]
                else:
                    curr_start[j + 1] = curr_start[j]
            elif score == diag:
                if prev_row[0, j] == 0:
                    curr_start[j + 1] = [i + 1, j + 1]
                else:
                    curr_start[j + 1] = prev_start[j]
            else:
                if prev_row[0, j + 1] == 0:
                    curr_start[j + 1] = [i + 1, j + 1]
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
        prev_start[0] = [i + 1, 0]
        curr_start = [[0, 0] for i in range(n + 1)]

    # Swap back if we swapped the sequences
    if swapped:
        best_start.reverse()
        best_end.reverse()
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp

    print("Best Start: ", best_start)
    print("Best End: ", best_end)

    new_sequence1 = sequence1[max(best_start[0] - 1, 0): best_end[0]]
    new_sequence2 = sequence2[max(best_start[1] - 1, 0): best_end[1]]

    # Start the recursion - TODO finish - need to trim the sequences passed
    return(dynproglin_recursive(alphabet, scoring_matrix, new_sequence1,
                         new_sequence2,
                         best_start))


def dynproglin_recursive(alphabet, scoring_matrix, sequence1, sequence2,
                         start):
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)
    best_start = [0, 0]
    best_score = 0
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

    mid_col = int(floor(n/2)) + 1  # TODO TEST THAT THIS WORKS

    # Base cases:
    if len(sequence2) == 2 and len(sequence1) == 2:
        return []

    if len(sequence2) == 2 and len(sequence1) == 1:
        return []

    if len(sequence2) == 1 and len(sequence1) == 2:
        return []

    if len(sequence1) == 1 and len(sequence2) == 1:
        a = 10
        return []

    # TODO - check if one off error in case of errors
    # For the middle values, we keep track of two numbers - when we enter
    # the middle column and when we exit the middle column
    prev_mid = [[0, 0] for i in range(n - mid_col + 1)]
    curr_mid = [[0, 0] for i in range(n - mid_col + 1)]

    prev_row = np.zeros((1, n + 1))
    curr_row = np.zeros((1, n + 1))

    # Keep track of any vertical movement on the start and end too
    prev_start = [i for i in range(n + 1)]  # TODO CHECK
    curr_start = [i for i in range(n + 1)]
    prev_end = 0
    curr_end = 0


    # TODO - make sure the first row/column are correct
    prev_row[0, 0] = scoring_matrix[index['_'], index['_']]
    for j, k in enumerate(sequence2):
        prev_row[0, j + 1] = prev_row[0, j] + scoring_matrix[index[k]][
            index['_']]

    # Rest of matrix:
    for i, l in enumerate(sequence1):
        # TODO - ADD THIS LINE TO THE LOCAL INITIAL RUN - CANNOT ASSUME THAT
        #  THE SCORING MATRIX WILL BE NICE
        curr_row[0, 0] = prev_row[0, 0] + scoring_matrix[index[l]][index['_']]
        for j, k in enumerate(sequence2):
            left = curr_row[0, j] + scoring_matrix[index['_'], index[k]]
            diag = prev_row[0, j] + scoring_matrix[index[l], index[k]]
            up = prev_row[0, j + 1] + scoring_matrix[index[l], index['_']]

            score = max(left, diag, up)  # Trying to find a global
            # alignment now, so no 0 included
            curr_row[0, j + 1] = score

            # We only need to keep track of the mid_value for the half of
            # the matrix after the mid_col
            if j >= mid_col - 1:
                # If we are currently on the previous column to the middle:
                # Keep track of when we joined the middle column
                if j == mid_col - 1:
                    if score == left:
                        curr_mid[0][0] = i + 1
                    elif score == diag:
                        curr_mid[0][0] = i
                    else:
                        curr_mid[0][0] = prev_mid[0][0]

                # If we are currently on the middle column
                # Keep track of where we leave the middle column
                elif j == mid_col:
                    # If it came from the left, then curr_mid[j + 1] = i + 1
                    # If it came from the diag, then curr_mid[j + 1] = i
                    # If it came from above, then curr_mid[j + 1] = prev_mid[i + 1]
                    if score == left:
                        curr_mid[1] = [curr_mid[0][0], i + 1]  # Make note of
                        # start and row of middle column that path traverses
                    elif score == diag:
                        curr_mid[1] = [prev_mid[0][0], i]
                    else:
                        curr_mid[1] = list(prev_mid[1])
                # If we're not on the middle column, then propagate the
                # middle along the path
                else:  # TODO THIS INDEXING MIGHT NOT BE CORRECT
                    if score == left:
                        curr_mid[j - mid_col + 1] = list(curr_mid[j - mid_col])
                    elif score == diag:
                        curr_mid[j - mid_col + 1] = list(prev_mid[j - mid_col])
                    else:
                        curr_mid[j - mid_col + 1] = list(prev_mid[j - mid_col
                                                                + 1])

            # Keep track of any end movement TODO CHECK THIS WORKS
            if j == n - 1:
                if score == up:
                    curr_end = prev_end
                else:
                    curr_end = i + 1

            # Keep track of any start movement: TODO CARRY ON HERE - not
            #  TODO sure this works
            if score == left:
                if j == 0:
                    curr_start[j + 1] = i + 1
                else:
                    curr_start[j + 1] = curr_start[j]
            elif score == diag:
                if j == 0:
                    curr_start[j + 1] = i
                else:
                    curr_start[j + 1] = prev_start[j]
            else:
                if j == 0:
                    curr_start[j + 1] = i + 1
                else:
                    curr_start[j + 1] = prev_start[j + 1]

            if score > best_score:
                best_score = score
                best_start = curr_start[j + 1]

        # TODO - check this
        # Reset at end of row
        prev_row = curr_row
        curr_row = np.zeros((1, n + 1))

        prev_mid = curr_mid
        prev_mid[0][0] = i + 1  # TODO - modify/check this
        curr_mid = [[0, 0] for i in range(n - mid_col + 1)]


        prev_start = curr_start
        curr_start = [i for i in range(n + 1)]

    a = 10
    # TODO - check this
    mid_points = []
    start_points = []
    end_points = []

    for i in range(1, prev_start[n] + 1):
        start_points.append([i, 1])

    for i in range(curr_end, m + 1):
        end_points.append([i, n])

    for i in range(prev_mid[n - mid_col][0], prev_mid[n - mid_col][1] + 1):
        mid_points.append([i, mid_col])

    # Swap back
    if swapped:
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp

        for point in start_points + mid_points + end_points:
            point.reverse()

    seq1_head = sequence1[start_points[-1][0] - 1:mid_points[0][0]]
    seq2_head = sequence2[start_points[-1][1] - 1:mid_points[0][1]]
    seq1_tail = sequence1[mid_points[-1][0] - 1:end_points[0][0]]
    seq2_tail = sequence2[mid_points[-1][1] - 1:end_points[0][1]]

    # Alter the mid point so that it is a midpoint of the total matrix,
    # not just one of the sub matrix
    real_mid_points = []
    real_start_points = []
    real_end_points = []

    for point in start_points:
        real_start_points.append([sum(x) - 1 for x in zip(start, point)])

    for point in end_points:
        real_end_points.append([sum(x) - 1 for x in zip(start, point)])

    for point in mid_points:
        real_mid_points.append([sum(x) - 1 for x in zip(start, point)])

    a = 10

    return real_start_points + dynproglin_recursive(
        alphabet, scoring_matrix, seq1_head,
        seq2_head, start
    ) + real_mid_points + dynproglin_recursive(
        alphabet, scoring_matrix, seq1_tail,
        seq2_tail, real_mid_points[-1]
    ) + real_end_points


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

    #dynprog(alphabet, scoring_matrix, seq1, seq2)
    results = dynprog(alphabet, scoring_matrix, seq1, seq2)
    print("Sequence 1: ", seq1)
    print("Sequence 2: ", seq2)
    print("Score: ", results[0])
    print("Sequence 1 Indices: ", results[1])
    print("Sequence 2 Indices: ", results[2])

    print(dynproglin(alphabet, scoring_matrix, seq1, seq2))


if __name__ == "__main__":
    main()
