import numpy as np
from math import floor, log
from itertools import product
import sys
import ast


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
                # ^ Start a new sub-sequence
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

    return [best_score, seq1_indices, seq2_indices]


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
        n = m
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
        curr_start = [[0, 0]] * (n + 1)

    # Swap back if we swapped the sequences
    if swapped:
        best_start.reverse()
        best_end.reverse()
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp

    # Recursively break the matrix in half, finding the midpoint through
    # which the optimal path passes through in that sub-matrix
    coords = [best_start] + dynproglin_recursive(
        alphabet,
        scoring_matrix,
        sequence1[max(best_start[0] - 1, 0):best_end[0]],
        sequence2[max(best_start[1] - 1, 0):best_end[1]],
        best_start
    ) + [best_end]

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

    # Base cases:
    if m == 1:
        if n > 2:
            return [[start[0], start[1] + 1 + i] for i in range(n - 2)]
        else:
            return []

    if n == 1:
        if m > 2:
            return [[start[0] + 1 + i, start[1]] for i in range(m - 2)]
        else:
            return []

    if n == 2 and m == 2:
        left = scoring_matrix[index[sequence1[0]], index[sequence2[0]]] + \
               scoring_matrix[index[sequence1[1]], index['_']] + \
               scoring_matrix[index['_'], index[sequence2[1]]]

        diag = scoring_matrix[index[sequence1[0]], index[sequence2[0]]] + \
            scoring_matrix[index[sequence1[1]], index[sequence2[1]]]

        up = scoring_matrix[index[sequence1[0]], index[sequence2[0]]] + \
            scoring_matrix[index['_'], index[sequence2[1]]] + \
            scoring_matrix[index[sequence1[1]], index['_']]

        score = max(left, diag, up)

        if score == left:
            return [[start[0] + 1, start[1]]]
        elif score == diag:
            return []
        else:
            return [[start[0], start[1] + 1]]

    # Ensure we use the smaller of the two to iterate through, reducing memory
    if m < n:
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        temp = m
        m = n
        n = temp
        swapped = True

    mid_col = int(floor(n / 2))

    # Used to keep track of on which row the middle column is joined and left
    prev_mid = [[0, 0]] * (n - mid_col)
    curr_mid = [[0, 0]] * (n - mid_col)

    prev_row = np.zeros((1, n))
    curr_row = np.zeros((1, n))

    # Used to keep track of on which row we leave the start & join the end cols
    prev_start = [0] * n
    prev_start[0] = 1
    curr_start = [0] * n
    curr_end = 0

    prev_row[0, 0] = scoring_matrix[index[sequence1[0]], index[sequence2[0]]]
    for j, k in enumerate(sequence2[1:]):
        prev_row[0, j + 1] = prev_row[0, j] + scoring_matrix[index[k]][
            index['_']]

    # Rest of matrix:
    for i in range(len(sequence1) - 1):
        l = sequence1[i + 1]
        curr_row[0, 0] = prev_row[0, 0] + scoring_matrix[index[l],
                                                         index['_']]

        curr_start[0] = i + 1
        for j in range(len(sequence2) - 1):
            k = sequence2[j + 1]

            left = curr_row[0, j] + scoring_matrix[index['_'], index[k]]
            diag = prev_row[0, j] + scoring_matrix[index[l], index[k]]
            up = prev_row[0, j + 1] + scoring_matrix[index[l], index['_']]

            score = max(left, diag, up)
            curr_row[0, j + 1] = score

            # Keep track of when we leave the start column
            if j == 0:
                if score == left:
                    curr_start[j + 1] = i + 1
                elif score == diag:
                    curr_start[j + 1] = i
                else:
                    curr_start[j + 1] = prev_start[j + 1]

            if j > 0:
                if score == left:
                    curr_start[j + 1] = curr_start[j]
                elif score == diag:
                    curr_start[j + 1] = prev_start[j]
                else:
                    curr_start[j + 1] = prev_start[j + 1]

            # If n == 2 and m > 2, we don't need to find midpoints
            if n > 2:
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
                            curr_mid[j - mid_col + 1] = list(
                                curr_mid[j - mid_col]
                            )
                        elif score == diag:
                            curr_mid[j - mid_col + 1] = list(
                                prev_mid[j - mid_col]
                            )
                        else:
                            curr_mid[j - mid_col + 1] = list(
                                prev_mid[j - mid_col + 1]
                            )

            # Keep track of when we join the final column
            if j == n - 2:
                if score == left or score == diag:
                    curr_end = i + 1

            # Keep track of best score
            if score > best_score:
                best_score = score

        # Reset at end of row
        prev_row = curr_row
        curr_row = np.zeros((1, n))
        prev_mid = curr_mid
        curr_mid = [[0, 0]] * (n - mid_col)
        prev_start = curr_start
        curr_start = [0] * n

    # Create the coordinates of the cells in the start, middle and end columns
    mid_points = []
    start_points = []
    end_points = []

    for i in range(prev_start[n - 1] + 1):
        start_points.append([i, 0])

    for i in range(curr_end, m):
        end_points.append([i, n - 1])

    # We only have midpoints if n > 2
    if n > 2:
        for i in range(prev_mid[n - mid_col - 1][0],
                       prev_mid[n - mid_col - 1][1] + 1):
            mid_points.append([i, mid_col])

    # If we swapped the sequences swap them back, and transpose all coordinates
    if swapped:
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp

        for point in start_points + mid_points + end_points:
            point.reverse()

    # Convert the local coordinates to coordinates of the whole matrix
    real_mid_points = []
    real_start_points = []
    real_end_points = []

    for point in start_points:
        real_start_points.append([sum(x) for x in zip(start, point)])

    for point in end_points:
        real_end_points.append([sum(x) for x in zip(start, point)])

    # We only have midpoints if n > 2
    if n > 2:
        for point in mid_points:
            real_mid_points.append([sum(x) for x in zip(start, point)])

    # If n == 2, then we end recursion:
    if n == 2:
        return real_start_points[1:] + real_end_points[:-1]

    # Split the two sequences into sub-sequences for the next recursive call
    seq1_head = sequence1[start_points[-1][0]:mid_points[0][0] + 1]
    seq2_head = sequence2[start_points[-1][1]:mid_points[0][1] + 1]
    seq1_tail = sequence1[mid_points[-1][0]:end_points[0][0] + 1]
    seq2_tail = sequence2[mid_points[-1][1]:end_points[0][1] + 1]

    # Find the midpoint of the smaller sub-matrices
    return real_start_points[1:] + dynproglin_recursive(
        alphabet, scoring_matrix, seq1_head,
        seq2_head, real_start_points[-1]
    ) + real_mid_points + dynproglin_recursive(
        alphabet, scoring_matrix, seq1_tail,
        seq2_tail, real_mid_points[-1]
    ) + real_end_points[:-1]


def dynprog_banded(alphabet, scoring_matrix, sequence1, sequence2, seeds,
                   swapped):
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)
    m = len(sequence1)
    n = len(sequence2)
    best_path_score = 0  # Holds the score of the best path in current DP iter
    best_align_matrix = []
    best_pointer_matrix = []
    best_final_score = 0  # Holds the final best score

    # Banded DP Parameters
    b = max(int(0.1 * len(sequence2)), 10)  # Width of the band TODO CHANGE
    # BACK
    h = 10  # Number of iterations of banded dp to run

    # Convert the seeds dict to a list of HSPs, sorted by score
    hsps = sorted(
        [hsp for hsp_list in seeds.values() for hsp in hsp_list],
        key=lambda hsp_tup: hsp_tup[2]
    )[-h:]

    for hsp in hsps:
        diag = hsp[0][1] - hsp[1][1]

        # TODO 3: Stop being lazy and improve the amount of space used
        align_matrix = np.zeros(shape=(m + 1, n + 1))
        pointers = np.zeros(shape=(m + 1, n + 1))

        # First row
        for j, k in enumerate(sequence2):
            score = max(
                0,
                align_matrix[0, j] + scoring_matrix[index[k]][index['_']]
            )

            if score == 0:
                pointers[0, j + 1] = 0
            else:
                pointers[0, j + 1] = 1

            align_matrix[0, j + 1] = score

        # First column
        for i, l in enumerate(sequence1):
            score = max(
                0,
                align_matrix[i, 0] + scoring_matrix[index[l]][index['_']]
            )

            if score == 0:
                pointers[i + 1, 0] = 0
            else:
                pointers[i + 1, 0] = 3

            align_matrix[i + 1, 0] = score

        # Rest of the matrix:
        for i, l in enumerate(sequence1):
            for j, k in enumerate(sequence2):
                i_range = range(
                    max(0, j - diag - b - 1),
                    min(m, j - diag + b + 1)
                )
                j_range = range(
                    max(0, i + diag - b - 1),
                    min(n, i + diag + b + 1)
                )

                # Continue only if in the band
                if i not in i_range or j not in j_range:
                    continue

                # On the top boundary of the band
                if i == (j - diag + b) and j == (i + diag + b):
                    scores = np.array([
                        0,
                        align_matrix[i + 1, j] + scoring_matrix[
                            index['_'], index[k]],
                        align_matrix[i, j] + scoring_matrix[
                            index[l], index[k]],
                    ])

                # On the bottom boundary of the band
                elif i == (j - diag - b) and j == (i + diag - b):
                    scores = np.array([
                        0,
                        align_matrix[i, j] + scoring_matrix[
                            index[l], index[k]],
                        align_matrix[i, j + 1] + scoring_matrix[
                            index[l], index['_']]
                    ])
                else:
                    scores = np.array([
                        0,
                        align_matrix[i + 1, j] + scoring_matrix[
                            index['_'], index[k]],
                        align_matrix[i, j] + scoring_matrix[
                            index[l], index[k]],
                        align_matrix[i, j + 1] + scoring_matrix[
                            index[l], index['_']]
                    ])

                score = scores.max()
                pointers[i + 1, j + 1] = scores.argmax()  # Poor style

                if score > best_path_score:
                    best_path_score = score

                align_matrix[i + 1, j + 1] = score

        if best_path_score > best_final_score:
            best_final_score = best_path_score
            best_align_matrix = align_matrix
            best_pointer_matrix = pointers

    # Form the indices
    i, j = np.unravel_index(
        best_align_matrix.argmax(),
        best_align_matrix.shape
    )
    direction = best_pointer_matrix[i, j]
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

        direction = best_pointer_matrix[i, j]

    seq1_indices.reverse()
    seq2_indices.reverse()

    if swapped:
        return [best_final_score, seq2_indices, seq1_indices]
    else:
        return [best_final_score, seq1_indices, seq2_indices]


def heuralign(alphabet, scoring_matrix, sequence1, sequence2):
    swapped = False
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)

    # BLAST Parameters
    k = 3  # Length of the k-words
    t = 10  # No. of high-scoring tuples to keep
    # TODO assume scores are normally distributed and choose only the top 5%
    e = 3  # Extension threshold

    # Swap sequences
    if len(sequence1) < len(sequence2):
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        swapped = True

    m = len(sequence1)
    n = len(sequence2)

    seeds = {}

    # This outer while loop is necessary, as for small sequences we may end
    # up in the case where a length of 3 for our k-words
    while len(seeds) == 0:  # TODO - check the base case of k = 0
        k_word_list = []
        # Generate the k-word list
        for i in range(n - k):
            k_word_list.append(sequence2[i:i + k])

        k_word_list = list(set(k_word_list))

        # Generate the word neighbours
        k_word_neighbours = {k: [] for k in k_word_list}
        k_tuples = [''.join(i) for i in product(alphabet, repeat=k)]
        for word in k_word_list:
            for tup in k_tuples:
                score = 0

                for i in range(k):
                    score += scoring_matrix[index[word[i]], index[tup[i]]]

                # TODO 1: See if removing this makes a difference
                if score <= 0:
                    continue

                k_word_neighbours[word].append((tup, score))

        # Keep only the best T k_word_neighbours of each k_tuple, then convert
        # each [('b', score1), ('c': score2)] into [('b', 'c'),
        # (score1, score2)]
        for key, val in k_word_neighbours.items():
            k_word_neighbours[key] = list(
                zip(*sorted(val, key=lambda tup_score: tup_score[1])[-t:])
            )

        # Generate the dict of seeds, indexed by (i - j). {(i - j): [[(i1, j1),
        # score1], [(i2, j2), score2], ... ], (i' - j'): [ ... ], ...}
        for j in range(n - k):
            for i in range(m - k):
                try:
                    query = sequence1[i: i + k]
                    target = sequence2[j: j + k]

                    k_word_list_score = k_word_neighbours[target]
                    k_word_list = k_word_list_score[0]

                    ind = k_word_list.index(query)
                    score = k_word_neighbours[target][1][ind]

                    seeds[j - i] = seeds.get(j - i, []) + [
                        [(i, j), (i + k - 1, j + k - 1), score]
                    ]

                    # TODO - keep track of highest scoring seed, and make sure
                    #  that we manually extend this one and dynamic program it,
                    #  in the case it is an isolated match on a diag which is
                    #  long

                except ValueError:
                    continue

        k -= 1

    k += 1
    # Extend the seeds along the diagonals with a sufficiently large number
    # of matches, giving us our High Scoring Pairs (HSPs)
    for key, val in seeds.items():
        if len(val) > e:
            for seed in val:
                left_score = 0
                right_score = 0
                score = 0

                # TODO 2: Add the check for a single gap heuristic
                # Try extend left:
                i, j = seed[0]
                while score >= 0 and i > 0 and j > 0:
                    score = scoring_matrix[
                        index[sequence1[i - 1]],
                        index[sequence2[j - 1]]
                    ]

                    i -= 1
                    j -= 1
                    left_score += score

                # New starting coordinates and left_score
                seed[0] = (i + 1, j + 1)
                left_score -= score

                # Try extend right:
                i, j = seed[1]
                score = 0
                while score >= 0 and i < m - 1 and j < n - 1:
                    score = scoring_matrix[
                        index[sequence1[i + 1]],
                        index[sequence2[j + 1]]
                    ]

                    i += 1
                    j += 1
                    right_score += score

                # New end coordinates and right_score
                seed[1] = (i - 1, j - 1)
                right_score -= score

                # Update overall score
                seed[2] = seed[2] + left_score + right_score

        else:
            # TODO - handle the case where there aren't any which pass the
            #  extension threshold
            continue
            # Delete diagonal from the

    return dynprog_banded(
        alphabet, scoring_matrix, sequence1, sequence2, seeds, swapped
    )


def dynprogcost(sequence1, sequence2):
    alphabet = ['A', 'B', 'C']
    index = {i: alphabet.index(i) for i in alphabet}
    best_score = 0
    m = len(sequence1)
    n = len(sequence2)

    # Score Function Parameters
    scoring_matrix = np.array([[1, -1, -2], [-1, 2, -4], [-2, -4, 3]])
    p = -5
    def c(t): return 1 + (0.1 * t) if 0 <= t <= 10 else 2

    align_matrix = np.zeros(shape=(m + 1, n + 1))
    pointers = np.zeros(shape=(m + 1, n + 1))

    # Rest of the matrix:
    for i, l in enumerate(sequence1):
        for j, k in enumerate(sequence2):

            # Back track left:
            a, b = i + 1, j
            left_gap = 0
            while pointers[a, b] == 1:
                left_gap += 1
                b -= 1
            left = align_matrix[a, b] + (p * log(left_gap + 2))

            # Back track up:
            up_gap = 0
            a, b = i, j + 1
            while pointers[a, b] == 3:
                up_gap += 1
                a -= 1
            up = align_matrix[a, b] + (p * log(up_gap + 2))

            # Codon score
            a, b = i, j
            matched = 0
            while pointers[a, b] == 2:
                matched += 1
                a -= 1
                b -= 1
            diag = align_matrix[i, j] + (
                    scoring_matrix[index[l], index[k]] * c(floor(matched/3.0))
            )

            scores = np.array([0, left, diag, up])
            score = scores.max()
            pointers[i + 1, j + 1] = scores.argmax()

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
    print(align_matrix)
    print(pointers)

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

    return [best_score, seq1_indices, seq2_indices]


def check_indices(alphabet, scoring_matrix, sequence1, sequence2, indices1,
                  indices2, calc_score):
    scoring_matrix = np.array(scoring_matrix)
    index = {i: alphabet.index(i) for i in alphabet}
    index['_'] = len(alphabet)

    score = 0
    x, y = indices1[0], indices2[0]

    for c in range(len(indices1)):
        i = indices1[c]
        j = indices2[c]

        while x < i:
            score += scoring_matrix[index[sequence1[x]], index['_']]
            x += 1

        while y < j:
            score += scoring_matrix[index[sequence2[y]], index['_']]
            y += 1

        score += scoring_matrix[index[sequence1[i]], index[sequence2[j]]]
        x += 1
        y += 1

    print(
        "Does the calculated score equal the actual score?",
        "Yes." if score == calc_score else "No."
    )


def main():
    # Parse Input
    alphabet = sys.argv[1]
    scoring_matrix = ast.literal_eval(sys.argv[2])
    seq1 = sys.argv[3]
    seq2 = sys.argv[4]

    # Print format options for numpy
    np.set_printoptions(edgeitems=20, linewidth=100000, precision=2)

    # Run the four algorithms
    quad_results = dynprog(alphabet, scoring_matrix, seq1, seq2)
    lin_results = dynproglin(alphabet, scoring_matrix, seq1, seq2)
    heur_results = heuralign(alphabet, scoring_matrix, seq1, seq2)
    own_results = dynprogcost(seq1, seq2)

    # Print results
    print("Sequence 1: ", seq1)
    print("Sequence 2: ", seq2)
    print("Score (Quadratic): ", quad_results[0])
    print("Score (Linear):    ", lin_results[0])
    print("Score (Heuristic): ", heur_results[0])
    print("Score (Own):       ", own_results[0])
    print("Sequence 1 Indices (Quadratic):", quad_results[1])
    print("Sequence 1 Indices (Linear):   ", lin_results[1])
    print("Sequence 1 Indices (Heuristic):", heur_results[1])
    print("Sequence 1 Indices (Own):      ", own_results[1])
    print("Sequence 2 Indices (Quadratic):", quad_results[2])
    print("Sequence 2 Indices (Linear):   ", lin_results[2])
    print("Sequence 2 Indices (Heuristic):", heur_results[2])
    print("Sequence 2 Indices (Own):      ", own_results[2])

    # check_indices(alphabet, scoring_matrix, seq1, seq2, own_results[1],
    #               own_results[2], own_results[0])

    # TODO - memory management for part 2 if time - deleting/reassigning
    # variables etc
    # add var_name = none for numpy arrays and other variables to tell
    # Python GC that the memory can be reused


if __name__ == "__main__":
    main()
